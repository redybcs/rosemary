//! A drawable smaller than the rectangle it is shown in, scaled up by a
//! reconstruction filter instead of a linear blit: SGSR1's edge-direction
//! variant, or RAVU-Zoom r2 anti-ringing with RCAS after it. `ROSEMARY_UPSCALER`
//! picks one (`sgsr-ed`, `ravu-ar`);
//! unset, or built without glslc, the blit stays as it was.
//!
//! The frame is copied into an image of the upscaler's own before it is read,
//! so nothing here holds a view of an image the guest owns and may free.

use std::sync::{Mutex, OnceLock};

use ash::vk;

use crate::gpu::{self, Gpu, GpuBuffer, GpuImage, ImageDescription};

mod data {
    include!(concat!(env!("OUT_DIR"), "/upscale.rs"));
    include!(concat!(env!("OUT_DIR"), "/ravu_size.rs"));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sgsr,
    Ravu,
}

fn kind() -> Option<Kind> {
    static KIND: OnceLock<Option<Kind>> = OnceLock::new();
    *KIND.get_or_init(|| {
        if !data::AVAILABLE {
            return None;
        }
        match std::env::var("ROSEMARY_UPSCALER").ok()?.as_str() {
            "sgsr-ed" => Some(Kind::Sgsr),
            "ravu-ar" => Some(Kind::Ravu),
            _ => None,
        }
    })
}

/// The push block every shader here declares: source size, target size, a
/// sharpness none of them reads any more, and whether to swap red and blue.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    src_size: [i32; 2],
    dst_size: [i32; 2],
    sharpness: f32,
    swap_rb: i32,
}

/// The images for one source and target size, rebuilt when either changes.
struct Target {
    source: (u32, u32),
    target: (u32, u32),
    input: GpuImage,
    /// RAVU's result before sharpening; SGSR sharpens as it scales.
    middle: Option<GpuImage>,
    output: GpuImage,
    main_set: vk::DescriptorSet,
    sharpen_set: vk::DescriptorSet,
    fresh: bool,
}

struct Upscaler {
    gpu: &'static Gpu,
    kind: Kind,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sharpen: vk::Pipeline,
    pool: vk::DescriptorPool,
    /// RAVU's weight and anti-ringing tables, and the staging they came from.
    luts: Option<[(GpuImage, GpuBuffer); 2]>,
    luts_uploaded: bool,
    target: Option<Target>,
}

// SAFETY: every handle here belongs to the device, and the one mapped pointer
// (a staging buffer's) is only touched while the lock below is held.
unsafe impl Send for Upscaler {}

static UPSCALERS: Mutex<Vec<Upscaler>> = Mutex::new(Vec::new());

/// Scale `from` in `source` up into `to` in `target` with the chosen filter.
/// False when there is none, or the copy is not an enlargement, which leaves
/// it to the caller's blit.
///
/// # Safety
///
/// As [`super::present::blit`]: recording into `command_buffer`, `source` in
/// `GENERAL`, `target` in `target_layout`.
pub(super) unsafe fn blit(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    source: vk::Image,
    from: [vk::Offset3D; 2],
    target: vk::Image,
    target_layout: vk::ImageLayout,
    to: [vk::Offset3D; 2],
) -> bool {
    let Some(kind) = kind() else {
        return false;
    };
    let size = |r: [vk::Offset3D; 2]| {
        (
            (r[1].x - r[0].x).unsigned_abs(),
            (r[1].y - r[0].y).unsigned_abs(),
        )
    };
    let (source_size, target_size) = (size(from), size(to));
    let enlarges = source_size.0 <= target_size.0
        && source_size.1 <= target_size.1
        && source_size != target_size
        && source_size.0 > 0
        && source_size.1 > 0;
    if !enlarges {
        return false;
    }
    let Some(gpu) = crate::vulkan::gpu_by_handle(device.handle()) else {
        return false;
    };
    let mut upscalers = UPSCALERS.lock().expect("upscalers");
    if !upscalers.iter().any(|u| std::ptr::eq(u.gpu, gpu)) {
        match Upscaler::new(gpu, kind) {
            Ok(upscaler) => upscalers.push(upscaler),
            Err(reason) => {
                gpu::trace(format!("upscaler unavailable: {reason}"));
                return false;
            }
        }
    }
    let upscaler = upscalers
        .iter_mut()
        .find(|u| std::ptr::eq(u.gpu, gpu))
        .unwrap();
    match unsafe {
        upscaler.record(
            command_buffer,
            source,
            from,
            target,
            target_layout,
            to,
            source_size,
            target_size,
        )
    } {
        Ok(()) => true,
        Err(reason) => {
            gpu::trace(format!("upscale failed, blitting instead: {reason}"));
            false
        }
    }
}

fn image(
    gpu: &Gpu,
    format: vk::Format,
    (width, height): (u32, u32),
    usage: u64,
) -> Result<GpuImage, String> {
    gpu.create_image(&ImageDescription {
        image_type: vk::ImageType::TYPE_2D,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        width,
        height,
        depth: 1,
        levels: 1,
        layers: 1,
        samples: 1,
        usage,
        cube: false,
        components: vk::ComponentMapping::default(),
    })
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

impl Upscaler {
    fn new(gpu: &'static Gpu, kind: Kind) -> Result<Self, String> {
        let device = &gpu.device;
        let fail = |what: &str, error: vk::Result| format!("{what}: {error:?}");
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        // SAFETY: well-formed create infos on this live device, here and below.
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(|e| fail("sampler", e))?;
        let binding = |index: u32, kind: vk::DescriptorType| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(index)
                .descriptor_type(kind)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [
            binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            binding(1, vk::DescriptorType::STORAGE_IMAGE),
            binding(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|e| fail("descriptor set layout", e))?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<Params>() as u32)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )
        }
        .map_err(|e| fail("pipeline layout", e))?;
        let pipeline_of = |spirv: &[u8]| -> Result<vk::Pipeline, String> {
            let words =
                ash::util::read_spv(&mut std::io::Cursor::new(spirv)).map_err(|e| e.to_string())?;
            let module = unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            }
            .map_err(|e| fail("shader module", e))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline =
                unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) }
                    .map_err(|(_, e)| fail("compute pipeline", e))?[0];
            unsafe { device.destroy_shader_module(module, None) };
            Ok(pipeline)
        };
        let (pipeline, sharpen) = match kind {
            Kind::Sgsr => (pipeline_of(data::SGSR)?, vk::Pipeline::null()),
            Kind::Ravu => (pipeline_of(data::RAVU_AR)?, pipeline_of(data::SHARPEN)?),
        };
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 6,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2,
            },
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| fail("descriptor pool", e))?;
        let luts = match kind {
            Kind::Sgsr => None,
            Kind::Ravu => {
                let table = |bytes: &[u8]| -> Result<(GpuImage, GpuBuffer), String> {
                    let image = image(
                        gpu,
                        vk::Format::R16G16B16A16_SFLOAT,
                        data::RAVU_LUT_SIZE,
                        gpu::USAGE_SHADER_READ,
                    )?;
                    let staging = gpu.create_buffer(bytes.len() as u64)?;
                    // SAFETY: a mapped buffer of exactly this length.
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.contents, bytes.len())
                    };
                    Ok((image, staging))
                };
                Some([table(data::RAVU_LUT)?, table(data::RAVU_LUT_AR)?])
            }
        };
        Ok(Self {
            gpu,
            kind,
            sampler,
            set_layout,
            layout,
            pipeline,
            sharpen,
            pool,
            luts,
            luts_uploaded: false,
            target: None,
        })
    }

    /// Images and descriptor sets for these sizes, replacing any for others
    /// once the queue has finished with them.
    fn target_for(&mut self, source: (u32, u32), target: (u32, u32)) -> Result<(), String> {
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.source == source && t.target == target)
        {
            return Ok(());
        }
        let gpu = self.gpu;
        if let Some(old) = self.target.take() {
            // A resize, never a frame: the same wait presentation makes when
            // it rebuilds a swapchain.
            gpu.wait_queue_idle();
            for image in [Some(old.input), old.middle, Some(old.output)]
                .into_iter()
                .flatten()
            {
                gpu.destroy_image(image);
            }
            // SAFETY: the queue is idle, so no set from this pool is in use.
            unsafe {
                gpu.device
                    .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            }
            .map_err(|e| format!("descriptor pool reset: {e:?}"))?;
        }
        let rgba = vk::Format::R8G8B8A8_UNORM;
        let input = image(gpu, rgba, source, gpu::USAGE_SHADER_READ)?;
        let storage = gpu::USAGE_SHADER_READ | gpu::USAGE_SHADER_WRITE;
        let middle = match self.kind {
            Kind::Ravu => Some(image(gpu, rgba, target, storage)?),
            Kind::Sgsr => None,
        };
        let output = image(gpu, rgba, target, storage)?;
        let layouts = [self.set_layout, self.set_layout];
        let sets = unsafe {
            gpu.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| format!("descriptor sets: {e:?}"))?;
        let sampled = |view: vk::ImageView| {
            [vk::DescriptorImageInfo {
                sampler: self.sampler,
                image_view: view,
                image_layout: vk::ImageLayout::GENERAL,
            }]
        };
        let written = |view: vk::ImageView| {
            [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: view,
                image_layout: vk::ImageLayout::GENERAL,
            }]
        };
        let first_out = middle.as_ref().unwrap_or(&output).view;
        let (main_src, main_dst) = (sampled(input.view), written(first_out));
        let luts = self
            .luts
            .as_ref()
            .map(|[a, b]| (sampled(a.0.view), sampled(b.0.view)));
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&main_src),
            vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&main_dst),
        ];
        if let Some((lut, lut_ar)) = luts.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[0])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(lut),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[0])
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(lut_ar),
            );
        }
        let sharpen_src = middle.as_ref().map(|m| sampled(m.view));
        let sharpen_dst = written(output.view);
        if let Some(sharpen_src) = sharpen_src.as_ref() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[1])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(sharpen_src),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[1])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&sharpen_dst),
            );
        }
        // SAFETY: every set, view and sampler named is live and this device's.
        unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
        self.target = Some(Target {
            source,
            target,
            input,
            middle,
            output,
            main_set: sets[0],
            sharpen_set: sets[1],
            fresh: true,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn record(
        &mut self,
        cb: vk::CommandBuffer,
        source: vk::Image,
        from: [vk::Offset3D; 2],
        target: vk::Image,
        target_layout: vk::ImageLayout,
        to: [vk::Offset3D; 2],
        source_size: (u32, u32),
        target_size: (u32, u32),
    ) -> Result<(), String> {
        self.target_for(source_size, target_size)?;
        let device = &self.gpu.device;
        let memory = |src_stage, src_access, dst_stage, dst_access| unsafe {
            device.cmd_pipeline_barrier(
                cb,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)],
                &[],
                &[],
            )
        };
        let to_general = |images: &[vk::Image], old: vk::ImageLayout| {
            let barriers: Vec<_> = images
                .iter()
                .map(|&image| {
                    vk::ImageMemoryBarrier::default()
                        .old_layout(old)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .dst_access_mask(
                            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresource_range(color_range())
                })
                .collect();
            unsafe {
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                )
            };
        };

        if !self.luts_uploaded {
            if let Some(luts) = self.luts.as_ref() {
                let images: Vec<_> = luts.iter().map(|(image, _)| image.image).collect();
                for (image, staging) in luts {
                    let barrier = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image.image)
                        .subresource_range(color_range());
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(super::present::whole_color())
                        .image_extent(vk::Extent3D {
                            width: image.width,
                            height: image.height,
                            depth: 1,
                        });
                    unsafe {
                        device.cmd_pipeline_barrier(
                            cb,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[barrier],
                        );
                        device.cmd_copy_buffer_to_image(
                            cb,
                            staging.buffer,
                            image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[region],
                        );
                    }
                }
                to_general(&images, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            }
            self.luts_uploaded = true;
        }

        let t = self.target.as_mut().expect("made above");
        if t.fresh {
            let images: Vec<_> = [Some(&t.input), t.middle.as_ref(), Some(&t.output)]
                .into_iter()
                .flatten()
                .map(|i| i.image)
                .collect();
            to_general(&images, vk::ImageLayout::UNDEFINED);
            t.fresh = false;
        }
        // Whatever read or wrote these images last frame is done before the
        // copy in overwrites them.
        memory(
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
        );
        let whole = |(w, h): (u32, u32)| {
            [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: w as i32,
                    y: h as i32,
                    z: 1,
                },
            ]
        };
        let copy = |src: vk::Image, src_offsets, dst: vk::Image, dst_layout, dst_offsets| {
            let region = vk::ImageBlit::default()
                .src_subresource(super::present::whole_color())
                .src_offsets(src_offsets)
                .dst_subresource(super::present::whole_color())
                .dst_offsets(dst_offsets);
            unsafe {
                device.cmd_blit_image(
                    cb,
                    src,
                    vk::ImageLayout::GENERAL,
                    dst,
                    dst_layout,
                    &[region],
                    vk::Filter::NEAREST,
                )
            };
        };
        copy(
            source,
            from,
            t.input.image,
            vk::ImageLayout::GENERAL,
            whole(source_size),
        );
        memory(
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        let params = Params {
            src_size: [source_size.0 as i32, source_size.1 as i32],
            dst_size: [target_size.0 as i32, target_size.1 as i32],
            sharpness: 1.0,
            swap_rb: 0,
        };
        // SAFETY: `Params` is plain old data of exactly the pushed size.
        let push = unsafe {
            std::slice::from_raw_parts(
                (&params as *const Params).cast::<u8>(),
                std::mem::size_of::<Params>(),
            )
        };
        let groups = (target_size.0.div_ceil(8), target_size.1.div_ceil(8));
        let dispatch = |pipeline: vk::Pipeline, set: vk::DescriptorSet| unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            device.cmd_dispatch(cb, groups.0, groups.1, 1);
        };
        dispatch(self.pipeline, t.main_set);
        if t.middle.is_some() {
            memory(
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
            );
            let sharpen_push = Params {
                src_size: params.dst_size,
                ..params
            };
            let push = unsafe {
                std::slice::from_raw_parts(
                    (&sharpen_push as *const Params).cast::<u8>(),
                    std::mem::size_of::<Params>(),
                )
            };
            unsafe {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.sharpen);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[t.sharpen_set],
                    &[],
                );
                device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
                device.cmd_dispatch(cb, groups.0, groups.1, 1);
            }
        }
        memory(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
        copy(
            t.output.image,
            whole(target_size),
            target,
            target_layout,
            to,
        );
        Ok(())
    }
}
