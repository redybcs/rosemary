use super::*;

/// What a dispatch's grid size counts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Grid {
    Threadgroups,
    /// Metal rounds the thread count up to whole threadgroups.
    Threads,
}

pub(super) unsafe extern "C" fn dispatch_threadgroups(
    receiver: Id,
    _sel: Sel,
    groups: MtlSize,
    per_group: MtlSize,
) {
    // SAFETY: the runtime sends this only to encoders of this crate's.
    unsafe { dispatch(receiver, groups, per_group, Grid::Threadgroups) };
}

pub(super) unsafe extern "C" fn dispatch_threads(
    receiver: Id,
    _sel: Sel,
    threads: MtlSize,
    per_group: MtlSize,
) {
    // SAFETY: the runtime sends this only to encoders of this crate's.
    unsafe { dispatch(receiver, threads, per_group, Grid::Threads) };
}

/// # Safety
/// `receiver` is an encoder of this crate's.
unsafe fn dispatch(receiver: Id, extent: MtlSize, per_group: MtlSize, grid: Grid) {
    // SAFETY: an encoder of this crate's.
    let encoder = unsafe { object::body::<Encoder>(receiver) };
    if encoder.kind != Kind::Compute || encoder.compute_pipeline.is_null() {
        return;
    }
    // SAFETY: only a compute pipeline state of this crate's is ever bound.
    let state = unsafe { object::body::<ComputePipeline>(encoder.compute_pipeline) };
    let size = [
        per_group.width.max(1) as u32,
        per_group.height.max(1) as u32,
        per_group.depth.max(1) as u32,
    ];
    let pipeline = match encoder.compute_variant {
        Some((object, cached, pipeline))
            if object == encoder.compute_pipeline && cached == size =>
        {
            pipeline
        }
        _ => match state.variant(size) {
            Ok(pipeline) => {
                encoder.compute_variant = Some((encoder.compute_pipeline, size, pipeline));
                pipeline
            }
            Err(reason) => {
                return gpu::trace(format!("compute pipeline creation failed: {reason}"));
            }
        },
    };
    let counts = if grid == Grid::Threads {
        [
            (extent.width.div_ceil(per_group.width.max(1))) as u32,
            (extent.height.div_ceil(per_group.height.max(1))) as u32,
            (extent.depth.div_ceil(per_group.depth.max(1))) as u32,
        ]
    } else {
        [
            extent.width as u32,
            extent.height as u32,
            extent.depth as u32,
        ]
    };
    encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
    if !encoder.bind_resources(
        &state.resources,
        &state.inline_samplers,
        &state.layouts,
        vk::PipelineBindPoint::COMPUTE,
    ) {
        return;
    }
    gpu::milestone("compute dispatch");
    stats::count(Counter::Dispatches);
    let command = encoder.command();
    // SAFETY: recording into this command buffer.
    let label = state.profile_label.as_ref().map(|label| {
        format!(
            "{label} [{}x{}x{} groups of {}x{}x{}]",
            counts[0], counts[1], counts[2], size[0], size[1], size[2]
        )
        .into()
    });
    encoder.order_dispatches();
    encoder.record_as(label, || unsafe {
        command
            .gpu
            .device
            .cmd_dispatch(command.command_buffer, counts[0], counts[1], counts[2])
    });
    // A serial encoder's next dispatch sees everything this one wrote.
    if !encoder.concurrent {
        encoder.ordered.set(Ordered::DispatchesOwed);
    }
}

/// `-memoryBarrierWithScope:`, and the forms which name resources or stages.
///
/// Only a compute encoder orders work here: a render encoder's barrier falls
/// inside a Vulkan render pass, where this runtime records none.
pub(super) unsafe extern "C" fn memory_barrier(receiver: Id, _sel: Sel) {
    // SAFETY: an encoder of this crate's.
    let encoder = unsafe { object::body::<Encoder>(receiver) };
    if encoder.kind == Kind::Compute && encoder.ordered.get() == Ordered::Not {
        encoder.ordered.set(Ordered::DispatchesOwed);
    }
}

impl Encoder {
    /// Order the next dispatch after the ones before it, where it must wait
    /// for them. The barrier is recorded only once another dispatch follows,
    /// so the encoder's last one is ordered by the full barrier at
    /// `endEncoding` alone. A compute encoder records nothing but dispatches,
    /// so only compute work waits.
    fn order_dispatches(&self) {
        if self.ordered.get() != Ordered::DispatchesOwed {
            return;
        }
        let command = self.command();
        let memory = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        // SAFETY: recording into this command buffer, outside a render pass.
        unsafe {
            command.gpu.device.cmd_pipeline_barrier(
                command.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[memory],
                &[],
                &[],
            )
        };
        self.ordered.set(Ordered::Dispatches);
        count_barrier();
    }
}
