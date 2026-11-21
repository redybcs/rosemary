use super::*;
use std::sync::Mutex;

/// Every `MTLBuffer` made and destroyed, with its
/// length, its `MTLResourceOptions` and how many are live.
#[inline(always)]
pub(super) fn tracing_buffers() -> bool {
    #[dynamic]
    static ON: bool = std::env::var_os("ROSEMARY_TRACE_BUFFERS").is_some();

    *ON
}

/// `MTLStorageModePrivate`, as `MTLResourceOptions` carries it in bits 4-7.
const STORAGE_MODE_PRIVATE: u64 = 2;
/// `MTLStorageModeManaged`.
const STORAGE_MODE_MANAGED: u64 = 1;

/// The device's own copy of a managed buffer. On an adapter without unified
/// memory, which is what this device reports, Metal keeps a managed buffer
/// twice: the CPU writes one and `-didModifyRange:` says what to carry to the
/// other, which the device reads from its own memory rather than across the
/// bus on every use.
pub(crate) struct Mirror {
    buffer: GpuBuffer,
    /// What of `buffer` is older than the CPU's copy, as `start..end`.
    stale: Mutex<Option<(u64, u64)>>,
}

/// Managed buffers with a stale range, keyed by their device, each retained
/// until a commit carries the range over.
static MODIFIED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Mark `start..end` of a managed buffer as written by the CPU.
fn mark_modified(receiver: Id, start: u64, end: u64) {
    // SAFETY: a buffer of this crate's.
    let body = unsafe { object::body::<Buffer>(receiver) };
    let Some(mirror) = &body.mirror else {
        return;
    };
    let end = end.min(body.inner.length);
    if start >= end {
        return;
    }
    let mut stale = mirror.stale.lock().expect("stale range");
    match stale.as_mut() {
        Some(range) => *range = (range.0.min(start), range.1.max(end)),
        None => {
            *stale = Some((start, end));
            MODIFIED.lock().expect("modified buffers").push((
                std::ptr::from_ref(body.gpu) as usize,
                objc::arc::retain(receiver) as usize,
            ));
        }
    }
}

/// The managed buffers of `gpu` with a stale range, each carrying the
/// reference taken when it was marked, which the caller now owns.
pub(crate) fn take_modified(gpu: &Gpu) -> Vec<Id> {
    let key = std::ptr::from_ref(gpu) as usize;
    let mut modified = MODIFIED.lock().expect("modified buffers");
    let (mine, others): (Vec<_>, Vec<_>) = std::mem::take(&mut *modified)
        .into_iter()
        .partition(|(owner, _)| *owner == key);
    *modified = others;
    mine.into_iter().map(|(_, buffer)| buffer as Id).collect()
}

/// `-[MTLBuffer didModifyRange:]`.
pub(super) unsafe extern "C" fn buffer_did_modify_range(
    receiver: Id,
    _sel: Sel,
    range: crate::encoder::NsRange,
) {
    mark_modified(
        receiver,
        range.location,
        range.location.saturating_add(range.length),
    );
}

pub(crate) fn new_buffer(gpu: &'static Gpu, length: usize, options: u64, fill: gpu::Fill) -> Id {
    if length == 0 {
        return std::ptr::null_mut();
    }
    crate::stats::peak(crate::stats::Counter::BufferPeakBytes, length as u64);
    if tracing_buffers() {
        eprintln!(
            "rosemary: Metal: newBuffer {length} options {options:#x} live {}",
            crate::live_buffers(1)
        );
    }
    let private = (options >> 4) & 0xf == STORAGE_MODE_PRIVATE;
    let host = match private {
        true => None,
        false => gpu.create_host_pages(length as u64),
    };
    let inner = match &host {
        // The device reaches these bytes by importing them when it uses them.
        Some(pages) => GpuBuffer {
            contents: pages.contents(),
            length: length as u64,
            ..GpuBuffer::none()
        },
        None => match crate::stats::timed(crate::stats::Counter::BufferMicros, || match private {
            true => gpu.create_private_buffer(length as u64, fill),
            false => match fill {
                gpu::Fill::Zero => gpu.create_buffer(length as u64),
                gpu::Fill::Overwritten => gpu.create_staging_buffer(length as u64),
            },
        }) {
            Ok(inner) => inner,
            Err(reason) => {
                gpu::report_once(format!(
                    "newBufferWithLength: {length} bytes failed: {reason}"
                ));
                return std::ptr::null_mut();
            }
        },
    };
    let buffer = wrap_buffer(gpu, inner, options, host);
    if !buffer.is_null() && (options >> 4) & 0xf == STORAGE_MODE_MANAGED {
        match gpu.create_private_buffer(length as u64, gpu::Fill::Overwritten) {
            Ok(copy) => {
                // SAFETY: a buffer this call just made.
                unsafe { object::body::<Buffer>(buffer) }.mirror = Some(Mirror {
                    buffer: copy,
                    stale: Mutex::new(None),
                });
                // The device's copy starts out as the CPU's, which the
                // guest may still fill before it first commits.
                mark_modified(buffer, 0, length as u64);
            }
            Err(reason) => gpu::trace(format!("managed buffer device copy failed: {reason}")),
        }
    }
    buffer
}

/// The `MTLBuffer` over a Vulkan buffer this crate made.
pub(crate) fn wrap_buffer(
    gpu: &'static Gpu,
    inner: GpuBuffer,
    options: u64,
    host: Option<gpu::HostPages>,
) -> Id {
    let length = inner.length as usize;
    crate::account(length as i64);
    if (options >> 4) & 0xf != STORAGE_MODE_PRIVATE {
        crate::add_shared_buffer_bytes(length as u64);
    }
    crate::stats::count(crate::stats::Counter::BuffersCreated);
    crate::stats::add(crate::stats::Counter::BufferBytes, length as u64);
    object::make(
        c"RosemaryMTLBuffer",
        Buffer {
            gpu,
            inner,
            options,
            host,
            mirror: None,
        },
    )
}

impl Buffer {
    pub(crate) fn host_pages(&self) -> Option<&gpu::HostPages> {
        self.host.as_ref()
    }

    /// The buffer the device sees all of this one through, for a use whose
    /// extent no call states.
    pub(crate) fn device_buffer(&self) -> vk::Buffer {
        if let Some(mirror) = &self.mirror {
            return mirror.buffer.buffer;
        }
        match &self.host {
            Some(pages) => self.gpu.host_buffer(pages),
            None => self.inner.buffer,
        }
    }

    /// The buffer and offset the device sees `offset..offset + size` of this
    /// one through, for a copy. The caller has checked the range is inside.
    pub(crate) fn device_range(&self, offset: u64, size: u64) -> (vk::Buffer, u64) {
        if let Some(mirror) = &self.mirror {
            return (mirror.buffer.buffer, offset);
        }
        self.host_range(offset, size)
    }

    /// The buffer and offset the device reaches the CPU's copy of
    /// `offset..offset + size` through.
    pub(crate) fn host_range(&self, offset: u64, size: u64) -> (vk::Buffer, u64) {
        match &self.host {
            Some(pages) => self.gpu.host_range(pages, offset, size),
            None => (self.inner.buffer, offset),
        }
    }

    /// A managed buffer's device copy and its length.
    pub(crate) fn device_copy(&self) -> Option<(vk::Buffer, u64)> {
        self.mirror
            .as_ref()
            .map(|mirror| (mirror.buffer.buffer, self.inner.length))
    }

    /// A managed buffer's device copy and its stale range, which is cleared:
    /// the caller carries it over.
    pub(crate) fn take_stale(&self) -> Option<(vk::Buffer, u64, u64)> {
        let mirror = self.mirror.as_ref()?;
        let (start, end) = mirror.stale.lock().expect("stale range").take()?;
        Some((mirror.buffer.buffer, start, end))
    }
}

pub(super) unsafe extern "C" fn buffer_contents(receiver: Id, _sel: Sel) -> *mut u8 {
    // SAFETY: a buffer of this crate's.
    unsafe { object::body::<Buffer>(receiver) }.inner.contents
}

pub(super) unsafe extern "C" fn buffer_length(receiver: Id, _sel: Sel) -> usize {
    // SAFETY: a buffer of this crate's.
    unsafe { object::body::<Buffer>(receiver) }.inner.length as usize
}

pub(super) unsafe extern "C" fn buffer_storage_mode(receiver: Id, _sel: Sel) -> u64 {
    // SAFETY: a buffer of this crate's.
    (unsafe { object::body::<Buffer>(receiver) }.options >> 4) & 0xf
}

/// `-[MTLBuffer dealloc]`.
///
/// Without this every buffer a guest makes is a `VkBuffer` and a range of a
/// heap block held for the life of the process. A game which streams its world
/// in makes gigabytes of them in a second, so the leak is not a slow one: it
/// exhausts the adapter's memory during the first minute of play.
///
/// The last reference is gone, so nothing can bind it again, and a command
/// buffer which did bind it holds a reference of its own until the work it was
/// recorded into has completed.
/// TEMPORARY: `ROSEMARY_METAL_LEAK=1` keeps every device resource alive, to tell
/// a fault caused by freeing one too early from every other kind.
#[inline(always)]
pub(crate) fn leaking() -> bool {
    #[dynamic]
    static LEAK: bool = std::env::var_os("ROSEMARY_METAL_LEAK").is_some();

    *LEAK
}

pub(super) unsafe extern "C" fn buffer_dealloc(receiver: Id, _sel: Sel) {
    // SAFETY: a buffer of this crate's.
    let buffer = unsafe { object::body::<Buffer>(receiver) };
    if tracing_buffers() {
        eprintln!(
            "rosemary: Metal: freeBuffer {} live {}",
            buffer.inner.length,
            crate::live_buffers(-1)
        );
    }
    crate::account(-(buffer.inner.length as i64));
    if !buffer.inner.private {
        crate::sub_shared_buffer_bytes(buffer.inner.length);
    }
    heap::leave(receiver);
    let inner = std::mem::replace(&mut buffer.inner, GpuBuffer::none());
    let mirror = buffer.mirror.take();
    if !leaking() {
        if let Some(mirror) = mirror {
            buffer.gpu.destroy_buffer(mirror.buffer);
        }
        match buffer.host.take() {
            Some(pages) => buffer.gpu.destroy_host_pages(pages),
            None => buffer.gpu.destroy_buffer(inner),
        }
    }
    objc::arc::finalize(receiver);
}
