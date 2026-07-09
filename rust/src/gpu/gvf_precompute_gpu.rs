//! GPU-accelerated GVF geometry precompute.
//!
//! Ports the once-per-DSM building ray-trace (`precompute_gvf_geometry`) onto a
//! single GPU compute dispatch. One thread handles one pixel and iterates over
//! all azimuths internally, marching precomputed ray shifts. Produces the same
//! outputs the CPU path caches: per-azimuth `blocking_distance` + `facesh`, and
//! the 5 reduced `cached_albnosh_*` channels.
//!
//! Shares `Arc<wgpu::Device>` and `Arc<wgpu::Queue>` with `ShadowGpuContext`.

use ndarray::ArrayView2;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// Ensures mapped staging buffers are always unmapped on scope exit.
struct MappedBufferGuard<'a> {
    buffer: &'a wgpu::Buffer,
}

impl<'a> MappedBufferGuard<'a> {
    fn new(buffer: &'a wgpu::Buffer) -> Self {
        Self { buffer }
    }
}

impl Drop for MappedBufferGuard<'_> {
    fn drop(&mut self) {
        self.buffer.unmap();
    }
}

// ── Uniform buffer (must match Params in gvf_precompute.wgsl) ─────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GvfPrecomputeParams {
    rows: u32,
    cols: u32,
    num_azimuths: u32,
    max_steps: u32,
    first: f32,
    second: f32,
    wall_albedo: f32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AzimuthMetaGpu {
    dir_mask: u32,
    shift_offset: u32,
    azimuth_rad: f32,
    _pad1: u32,
}

/// Raw GPU output — read back and assembled into a `GvfGeometryCache`.
pub struct GvfPrecomputeRaw {
    /// 5 reduced channels, flat `[5 x rows x cols]`: center, E, S, W, N.
    pub albnosh: Vec<f32>,
    /// Per-azimuth blocking distance, flat `[num_azimuths x rows x cols]`.
    pub blocking_distance: Vec<u32>,
    /// Per-azimuth facesh mask, flat `[num_azimuths x rows x cols]`.
    pub facesh: Vec<f32>,
}

const NUM_ALBNOSH_CHANNELS: usize = 5;

// ── Cached GPU buffers ────────────────────────────────────────────────────

struct CachedBuffers {
    rows: usize,
    cols: usize,
    num_azimuths: usize,
    max_steps: usize,
    // Bind group 0
    params_buffer: wgpu::Buffer,
    azimuth_meta_buffer: wgpu::Buffer,
    shifts_buffer: wgpu::Buffer,
    // Bind group 1 (static per-DSM inputs)
    buildings_buffer: wgpu::Buffer,
    alb_buffer: wgpu::Buffer,
    aspect_buffer: wgpu::Buffer,
    wall_ht_buffer: wgpu::Buffer,
    // Bind group 2 (outputs)
    albnosh_buffer: wgpu::Buffer,
    bd_buffer: wgpu::Buffer,
    facesh_buffer: wgpu::Buffer,
    // Combined staging for readback
    staging_buffer: wgpu::Buffer,
    // Byte offsets/sizes within the staging buffer
    albnosh_bytes: u64,
    bd_bytes: u64,
    facesh_bytes: u64,
    // Bind groups
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    bind_group_2: wgpu::BindGroup,
}

// ── Public context ─────────────────────────────────────────────────────────

pub struct GvfPrecomputeGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_compute_workgroups_per_dimension: u32,
    pipeline: wgpu::ComputePipeline,
    bg_layout_0: wgpu::BindGroupLayout,
    bg_layout_1: wgpu::BindGroupLayout,
    bg_layout_2: wgpu::BindGroupLayout,
    cached: Mutex<Option<CachedBuffers>>,
}

impl GvfPrecomputeGpuContext {
    /// Create a new context, sharing device/queue from the shadow GPU context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GVF Precompute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gvf_precompute.wgsl").into()),
        });

        let bg_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG0 Layout"),
            entries: &Self::bg0_layout_entries(),
        });
        let bg_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG1 Layout"),
            entries: &Self::storage_ro_layout(4),
        });
        let bg_layout_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG2 Layout"),
            entries: &Self::storage_rw_layout(3),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GVF Precompute Pipeline Layout"),
            bind_group_layouts: &[&bg_layout_0, &bg_layout_1, &bg_layout_2],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GVF Precompute Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let max_compute_workgroups_per_dimension =
            device.limits().max_compute_workgroups_per_dimension;

        Self {
            device,
            queue,
            max_compute_workgroups_per_dimension,
            pipeline,
            bg_layout_0,
            bg_layout_1,
            bg_layout_2,
            cached: Mutex::new(None),
        }
    }

    /// Run the full precompute for one DSM and read back the raw outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn precompute(
        &self,
        buildings: ArrayView2<f32>,
        wall_aspect: ArrayView2<f32>,
        wall_ht: ArrayView2<f32>,
        alb_grid: ArrayView2<f32>,
        azimuths_deg: &[f32],
        shifts: &[Vec<(isize, isize)>],
        max_steps: usize,
        first: f32,
        second: f32,
        wall_albedo: f32,
    ) -> Result<GvfPrecomputeRaw, String> {
        let (rows, cols) = (buildings.nrows(), buildings.ncols());
        let num_azimuths = azimuths_deg.len();
        if num_azimuths == 0 {
            return Err("No azimuths provided to GVF precompute".to_string());
        }
        if max_steps == 0 {
            return Err("max_steps is zero in GVF precompute".to_string());
        }

        let mut cache = self.cached.lock().unwrap_or_else(|e| {
            eprintln!("WARNING: GVF precompute GPU cache mutex was poisoned, recovering");
            let mut guard = e.into_inner();
            *guard = None;
            guard
        });

        self.ensure_buffers_locked(&mut cache, rows, cols, num_azimuths, max_steps)?;
        let buffers = cache
            .as_ref()
            .ok_or_else(|| "GVF precompute GPU buffers missing after allocation".to_string())?;

        // ── Upload params ──
        let params = GvfPrecomputeParams {
            rows: rows as u32,
            cols: cols as u32,
            num_azimuths: num_azimuths as u32,
            max_steps: max_steps as u32,
            first,
            second,
            wall_albedo,
            _pad0: 0,
        };
        self.queue
            .write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

        // ── Upload azimuth metadata ──
        let mut meta_data = Vec::with_capacity(num_azimuths);
        for (i, &az) in azimuths_deg.iter().enumerate() {
            let mut dir_mask = 0u32;
            if (0.0..180.0).contains(&az) {
                dir_mask |= 1;
            } // E
            if (90.0..270.0).contains(&az) {
                dir_mask |= 2;
            } // S
            if (180.0..360.0).contains(&az) {
                dir_mask |= 4;
            } // W
            if !(90.0..270.0).contains(&az) {
                dir_mask |= 8;
            } // N
            meta_data.push(AzimuthMetaGpu {
                dir_mask,
                shift_offset: (i * max_steps) as u32,
                azimuth_rad: az * (std::f32::consts::PI / 180.0),
                _pad1: 0,
            });
        }
        self.queue.write_buffer(
            &buffers.azimuth_meta_buffer,
            0,
            bytemuck::cast_slice(&meta_data),
        );

        // ── Upload shifts (flattened, vec2<i32>) ──
        let total_shifts = num_azimuths * max_steps;
        let mut shift_data = vec![[0i32; 2]; total_shifts];
        for (i, az_shifts) in shifts.iter().enumerate() {
            for (n, &(dx, dy)) in az_shifts.iter().enumerate() {
                shift_data[i * max_steps + n] = [dx as i32, dy as i32];
            }
        }
        self.queue
            .write_buffer(&buffers.shifts_buffer, 0, bytemuck::cast_slice(&shift_data));

        // ── Upload static inputs ──
        Self::write_2d_f32(&self.queue, &buffers.buildings_buffer, &buildings);
        Self::write_2d_f32(&self.queue, &buffers.alb_buffer, &alb_grid);
        Self::write_2d_f32(&self.queue, &buffers.aspect_buffer, &wall_aspect);
        Self::write_2d_f32(&self.queue, &buffers.wall_ht_buffer, &wall_ht);

        // ── Dispatch ──
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GVF Precompute Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GVF Precompute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group_0, &[]);
            pass.set_bind_group(1, &buffers.bind_group_1, &[]);
            pass.set_bind_group(2, &buffers.bind_group_2, &[]);
            let (wg_x, wg_y) = self.checked_workgroups_2d(rows, cols, 8, 8, "GVF precompute")?;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ── Copy outputs into the combined staging buffer ──
        let mut offset = 0u64;
        encoder.copy_buffer_to_buffer(
            &buffers.albnosh_buffer,
            0,
            &buffers.staging_buffer,
            offset,
            buffers.albnosh_bytes,
        );
        offset += buffers.albnosh_bytes;
        encoder.copy_buffer_to_buffer(
            &buffers.bd_buffer,
            0,
            &buffers.staging_buffer,
            offset,
            buffers.bd_bytes,
        );
        offset += buffers.bd_bytes;
        encoder.copy_buffer_to_buffer(
            &buffers.facesh_buffer,
            0,
            &buffers.staging_buffer,
            offset,
            buffers.facesh_bytes,
        );

        let staging_size = buffers.albnosh_bytes + buffers.bd_bytes + buffers.facesh_bytes;
        let submission_index = self.queue.submit(Some(encoder.finish()));

        // ── Map + read back ──
        let buffer_slice = buffers.staging_buffer.slice(..staging_size);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| format!("GPU poll failed: {:?}", e))?;
        receiver
            .recv()
            .map_err(|e| format!("Channel recv failed: {}", e))?
            .map_err(|e| format!("Failed to map staging buffer: {:?}", e))?;

        let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
        let data = buffer_slice.get_mapped_range();

        let albnosh_len = (buffers.albnosh_bytes / 4) as usize;
        let bd_len = (buffers.bd_bytes / 4) as usize;
        let facesh_len = (buffers.facesh_bytes / 4) as usize;

        let a_end = buffers.albnosh_bytes as usize;
        let b_end = a_end + buffers.bd_bytes as usize;
        let f_end = b_end + buffers.facesh_bytes as usize;

        let albnosh: Vec<f32> = bytemuck::cast_slice(&data[0..a_end]).to_vec();
        let blocking_distance: Vec<u32> = bytemuck::cast_slice(&data[a_end..b_end]).to_vec();
        let facesh: Vec<f32> = bytemuck::cast_slice(&data[b_end..f_end]).to_vec();

        debug_assert_eq!(albnosh.len(), albnosh_len);
        debug_assert_eq!(blocking_distance.len(), bd_len);
        debug_assert_eq!(facesh.len(), facesh_len);

        Ok(GvfPrecomputeRaw {
            albnosh,
            blocking_distance,
            facesh,
        })
    }

    // ── Buffer management ─────────────────────────────────────────────────

    fn ensure_buffers_locked(
        &self,
        cache: &mut Option<CachedBuffers>,
        rows: usize,
        cols: usize,
        num_azimuths: usize,
        max_steps: usize,
    ) -> Result<(), String> {
        if let Some(ref c) = *cache {
            if c.rows == rows
                && c.cols == cols
                && c.num_azimuths == num_azimuths
                && c.max_steps == max_steps
            {
                return Ok(());
            }
        }

        let total_pixels = rows * cols;
        let pixel_bytes = (total_pixels * 4) as u64;
        let geom_pixels = num_azimuths * total_pixels;
        let albnosh_bytes = (NUM_ALBNOSH_CHANNELS * total_pixels * 4) as u64;
        let bd_bytes = (geom_pixels * 4) as u64;
        let facesh_bytes = (geom_pixels * 4) as u64;
        let total_shifts = num_azimuths * max_steps;
        let shifts_bytes = (total_shifts * 8) as u64; // vec2<i32> = 8 bytes
        let meta_bytes = (num_azimuths * std::mem::size_of::<AzimuthMetaGpu>()) as u64;

        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let input = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let output = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

        let params_buffer = make(
            "GVF Precompute Params",
            std::mem::size_of::<GvfPrecomputeParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let azimuth_meta_buffer = make("GVF Precompute AzimuthMeta", meta_bytes, input);
        let shifts_buffer = make("GVF Precompute Shifts", shifts_bytes, input);

        let buildings_buffer = make("GVF Precompute Buildings", pixel_bytes, input);
        let alb_buffer = make("GVF Precompute Albedo", pixel_bytes, input);
        let aspect_buffer = make("GVF Precompute Aspect", pixel_bytes, input);
        let wall_ht_buffer = make("GVF Precompute WallHt", pixel_bytes, input);

        let albnosh_buffer = make("GVF Precompute Albnosh", albnosh_bytes, output);
        let bd_buffer = make("GVF Precompute BlockingDist", bd_bytes, output);
        let facesh_buffer = make("GVF Precompute Facesh", facesh_bytes, output);

        let staging_buffer = make(
            "GVF Precompute Staging",
            albnosh_bytes + bd_bytes + facesh_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let err = pollster::block_on(self.device.pop_error_scope());
        if let Some(e) = err {
            *cache = None;
            return Err(format!(
                "GPU OOM allocating GVF precompute buffers for {}x{} grid ({} azimuths): {}",
                rows, cols, num_azimuths, e
            ));
        }

        let bind_group_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG0"),
            layout: &self.bg_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: azimuth_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shifts_buffer.as_entire_binding(),
                },
            ],
        });
        let bind_group_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG1"),
            layout: &self.bg_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buildings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: alb_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: aspect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wall_ht_buffer.as_entire_binding(),
                },
            ],
        });
        let bind_group_2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG2"),
            layout: &self.bg_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: albnosh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bd_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: facesh_buffer.as_entire_binding(),
                },
            ],
        });

        *cache = Some(CachedBuffers {
            rows,
            cols,
            num_azimuths,
            max_steps,
            params_buffer,
            azimuth_meta_buffer,
            shifts_buffer,
            buildings_buffer,
            alb_buffer,
            aspect_buffer,
            wall_ht_buffer,
            albnosh_buffer,
            bd_buffer,
            facesh_buffer,
            staging_buffer,
            albnosh_bytes,
            bd_bytes,
            facesh_bytes,
            bind_group_0,
            bind_group_1,
            bind_group_2,
        });
        Ok(())
    }

    fn checked_workgroups_2d(
        &self,
        rows: usize,
        cols: usize,
        workgroup_x: u32,
        workgroup_y: u32,
        label: &str,
    ) -> Result<(u32, u32), String> {
        let wg_x = (cols as u32).div_ceil(workgroup_x);
        let wg_y = (rows as u32).div_ceil(workgroup_y);
        let limit = self.max_compute_workgroups_per_dimension;
        if wg_x > limit || wg_y > limit {
            return Err(format!(
                "{} dispatch exceeds GPU workgroup limit {}: ({}, {}) for {}x{} grid",
                label, limit, wg_x, wg_y, rows, cols
            ));
        }
        Ok((wg_x, wg_y))
    }

    fn write_2d_f32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView2<f32>) {
        if let Some(slice) = arr.as_slice() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        } else {
            let contiguous: Vec<f32> = arr.iter().copied().collect();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&contiguous));
        }
    }

    // ── Bind group layouts ────────────────────────────────────────────────

    fn bg0_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            Self::storage_ro_entry(1),
            Self::storage_ro_entry(2),
        ]
    }

    fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_ro_layout(count: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        (0..count).map(Self::storage_ro_entry).collect()
    }

    fn storage_rw_layout(count: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        (0..count).map(Self::storage_rw_entry).collect()
    }
}
