//! GPU-accelerated GVF: geometry precompute + per-timestep thermal accumulation.
//!
//! One context owns the geometry buffers (`blocking_distance`, `facesh`). The
//! precompute shader (`gvf_precompute.wgsl`) writes them once per DSM directly
//! into those resident buffers; the per-timestep shader (`gvf_cached.wgsl`)
//! reads them in place across every timestep. The geometry is mirrored back to
//! the CPU exactly once (for the CPU fallback path and Python inspection) but is
//! never re-uploaded to the GPU — the old two-context design read it back and
//! re-flattened + re-uploaded it into a second buffer every tile, which is the
//! round trip this merge removes.
//!
//! `azimuth_meta` and `shifts` are shared by both pipelines (identical layout;
//! the per-timestep shader ignores the precompute-only `azimuth_rad` slot).
//!
//! Shares `Arc<wgpu::Device>` and `Arc<wgpu::Queue>` with `ShadowGpuContext`.

use ndarray::{Array2, ArrayView2};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::gvf_geometry::GvfGeometryCache;

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

// ── Per-timestep uniform (must match Params in gvf_cached.wgsl) ───────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GvfParams {
    rows: u32,
    cols: u32,
    num_azimuths: u32,
    max_steps: u32,
    first: f32,
    second: f32,
    lwall: f32,
    wall_albedo: f32,
}

// ── Precompute uniform (must match Params in gvf_precompute.wgsl) ─────────

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

// ── Shared azimuth metadata ──────────────────────────────────────────────
//
// One buffer feeds both shaders. The precompute shader reads `azimuth_rad`;
// the per-timestep shader declares that slot as padding and ignores it. Both
// read `dir_mask` (@0) and `shift_offset` (@4), which are at identical offsets.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AzimuthMetaGpu {
    dir_mask: u32,
    shift_offset: u32,
    azimuth_rad: f32,
    _pad1: u32,
}

const NUM_OUTPUT_CHANNELS: usize = 10;
const NUM_ALBNOSH_CHANNELS: usize = 5;

/// Raw precompute readback — the CPU mirror of the resident geometry.
///
/// `albnosh` is the 5 reduced channels `[5 x R x C]`; `blocking_distance` and
/// `facesh` are per-azimuth `[num_azimuths x R x C]`. Assembled into a
/// `GvfGeometryCache` by the caller.
pub struct GvfPrecomputeRaw {
    pub albnosh: Vec<f32>,
    pub blocking_distance: Vec<u32>,
    pub facesh: Vec<f32>,
}

// ── Cached GPU buffers ───────────────────────────────────────────────────

struct CachedBuffers {
    rows: usize,
    cols: usize,
    num_azimuths: usize,
    max_steps: usize,

    // Uniforms (one per pipeline)
    gvf_params_buffer: wgpu::Buffer,
    precompute_params_buffer: wgpu::Buffer,
    // Shared bind-group-0 storage inputs
    azimuth_meta_buffer: wgpu::Buffer,
    shifts_buffer: wgpu::Buffer,

    // Shared geometry (written by precompute, read by per-timestep)
    blocking_distance_buffer: wgpu::Buffer,
    facesh_buffer: wgpu::Buffer,

    // Precompute-only inputs + reduced output
    buildings_buffer: wgpu::Buffer,
    alb_buffer: wgpu::Buffer,
    aspect_buffer: wgpu::Buffer,
    wall_ht_buffer: wgpu::Buffer,
    albnosh_buffer: wgpu::Buffer,

    // Per-timestep inputs + output
    lup_buffer: wgpu::Buffer,
    albshadow_buffer: wgpu::Buffer,
    sunwall_mask_buffer: wgpu::Buffer,
    outputs_buffer: wgpu::Buffer,

    // Staging for readback
    staging_buffer: wgpu::Buffer,          // per-timestep outputs (10ch)
    readback_staging_buffer: wgpu::Buffer, // precompute albnosh + bd + facesh

    // Byte sizes within the precompute readback staging buffer
    albnosh_bytes: u64,
    bd_bytes: u64,
    facesh_bytes: u64,

    // Bind groups
    bind_group_0: wgpu::BindGroup, // per-timestep: params + meta + shifts
    bind_group_1: wgpu::BindGroup, // per-timestep: bd + facesh (read-only)
    bind_group_2: wgpu::BindGroup, // per-timestep: lup/albshadow/sunwall + outputs
    precompute_bind_group_0: wgpu::BindGroup, // precompute params + meta + shifts
    precompute_bind_group_1: wgpu::BindGroup, // precompute inputs
    precompute_bind_group_2: wgpu::BindGroup, // precompute outputs (albnosh/bd/facesh rw)

    // Track whether geometry is resident + ready for per-timestep dispatch
    geometry_uploaded: bool,
    readback_inflight: bool,
}

// ── Public context ───────────────────────────────────────────────────────

pub struct GvfGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_compute_workgroups_per_dimension: u32,
    pipeline: wgpu::ComputePipeline,
    precompute_pipeline: wgpu::ComputePipeline,
    bg_layout_0: wgpu::BindGroupLayout,
    bg_layout_1: wgpu::BindGroupLayout,
    bg_layout_2: wgpu::BindGroupLayout,
    bg_layout_p0: wgpu::BindGroupLayout,
    bg_layout_p1: wgpu::BindGroupLayout,
    bg_layout_p2: wgpu::BindGroupLayout,
    cached: Mutex<Option<CachedBuffers>>,
}

/// Raw GPU output — 10 accumulated arrays before scaling/baseline.
pub struct GvfGpuResult {
    pub lup: Array2<f32>,
    pub alb: Array2<f32>,
    pub lup_e: Array2<f32>,
    pub alb_e: Array2<f32>,
    pub lup_s: Array2<f32>,
    pub alb_s: Array2<f32>,
    pub lup_w: Array2<f32>,
    pub alb_w: Array2<f32>,
    pub lup_n: Array2<f32>,
    pub alb_n: Array2<f32>,
}

/// In-flight GPU dispatch token.
pub struct GvfGpuPending {
    rows: usize,
    cols: usize,
    total_pixels: usize,
    staging_size: u64,
    submission_index: wgpu::SubmissionIndex,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl GvfGpuContext {
    /// Create a new context, sharing device/queue from the shadow GPU context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GVF Cached Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gvf_cached.wgsl").into()),
        });
        let precompute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GVF Precompute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gvf_precompute.wgsl").into()),
        });

        // Per-timestep layouts
        let bg_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG0 Layout"),
            entries: &Self::bg0_layout_entries(),
        });
        let bg_layout_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG1 Layout"),
            entries: &Self::bg1_layout_entries(),
        });
        let bg_layout_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF BG2 Layout"),
            entries: &Self::bg2_layout_entries(),
        });

        // Precompute layouts
        let bg_layout_p0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG0 Layout"),
            entries: &Self::bg0_layout_entries(),
        });
        let bg_layout_p1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG1 Layout"),
            entries: &Self::storage_ro_layout(4),
        });
        let bg_layout_p2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GVF Precompute BG2 Layout"),
            entries: &Self::storage_rw_layout(3),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GVF Pipeline Layout"),
            bind_group_layouts: &[&bg_layout_0, &bg_layout_1, &bg_layout_2],
            push_constant_ranges: &[],
        });
        let precompute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GVF Precompute Pipeline Layout"),
                bind_group_layouts: &[&bg_layout_p0, &bg_layout_p1, &bg_layout_p2],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GVF Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let precompute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GVF Precompute Compute Pipeline"),
                layout: Some(&precompute_pipeline_layout),
                module: &precompute_shader,
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
            precompute_pipeline,
            bg_layout_0,
            bg_layout_1,
            bg_layout_2,
            bg_layout_p0,
            bg_layout_p1,
            bg_layout_p2,
            cached: Mutex::new(None),
        }
    }

    /// Run the geometry precompute on the GPU, writing `blocking_distance` and
    /// `facesh` into the resident geometry buffers (ready for per-timestep
    /// dispatch with no re-upload) and reading `albnosh` + the geometry back to
    /// the CPU as the fallback mirror. Also uploads the shared `azimuth_meta` +
    /// `shifts` so the per-timestep pipeline is fully prepared afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn precompute_geometry(
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
            eprintln!("WARNING: GVF GPU cache mutex was poisoned, recovering");
            let mut guard = e.into_inner();
            *guard = None;
            guard
        });

        self.ensure_buffers_locked(&mut cache, rows, cols, num_azimuths, max_steps)?;
        let buffers = cache
            .as_mut()
            .ok_or_else(|| "GVF GPU buffers missing after allocation".to_string())?;

        // ── Upload precompute params ──
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
        self.queue.write_buffer(
            &buffers.precompute_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // ── Upload shared azimuth metadata (precompute layout; the
        //    per-timestep shader ignores the azimuth_rad slot) ──
        Self::write_azimuth_meta(&self.queue, &buffers.azimuth_meta_buffer, azimuths_deg, max_steps);
        Self::write_shifts(&self.queue, &buffers.shifts_buffer, shifts, max_steps);

        // ── Upload precompute inputs ──
        Self::write_2d_f32(&self.queue, &buffers.buildings_buffer, &buildings);
        Self::write_2d_f32(&self.queue, &buffers.alb_buffer, &alb_grid);
        Self::write_2d_f32(&self.queue, &buffers.aspect_buffer, &wall_aspect);
        Self::write_2d_f32(&self.queue, &buffers.wall_ht_buffer, &wall_ht);

        // ── Dispatch precompute → resident bd/facesh + albnosh ──
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
            pass.set_pipeline(&self.precompute_pipeline);
            pass.set_bind_group(0, &buffers.precompute_bind_group_0, &[]);
            pass.set_bind_group(1, &buffers.precompute_bind_group_1, &[]);
            pass.set_bind_group(2, &buffers.precompute_bind_group_2, &[]);
            let (wg_x, wg_y) = self.checked_workgroups_2d(rows, cols, 8, 8, "GVF precompute")?;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // ── Copy outputs into the combined readback staging buffer ──
        let mut offset = 0u64;
        encoder.copy_buffer_to_buffer(
            &buffers.albnosh_buffer,
            0,
            &buffers.readback_staging_buffer,
            offset,
            buffers.albnosh_bytes,
        );
        offset += buffers.albnosh_bytes;
        encoder.copy_buffer_to_buffer(
            &buffers.blocking_distance_buffer,
            0,
            &buffers.readback_staging_buffer,
            offset,
            buffers.bd_bytes,
        );
        offset += buffers.bd_bytes;
        encoder.copy_buffer_to_buffer(
            &buffers.facesh_buffer,
            0,
            &buffers.readback_staging_buffer,
            offset,
            buffers.facesh_bytes,
        );

        let staging_size = buffers.albnosh_bytes + buffers.bd_bytes + buffers.facesh_bytes;
        let submission_index = self.queue.submit(Some(encoder.finish()));

        // ── Map + read back ──
        let buffer_slice = buffers.readback_staging_buffer.slice(..staging_size);
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

        let _unmap_guard = MappedBufferGuard::new(&buffers.readback_staging_buffer);
        let data = buffer_slice.get_mapped_range();

        let a_end = buffers.albnosh_bytes as usize;
        let b_end = a_end + buffers.bd_bytes as usize;
        let f_end = b_end + buffers.facesh_bytes as usize;

        let albnosh: Vec<f32> = bytemuck::cast_slice(&data[0..a_end]).to_vec();
        let blocking_distance: Vec<u32> = bytemuck::cast_slice(&data[a_end..b_end]).to_vec();
        let facesh: Vec<f32> = bytemuck::cast_slice(&data[b_end..f_end]).to_vec();

        // Geometry is now resident on the GPU + the shared meta/shifts are
        // uploaded, so the per-timestep pipeline is fully prepared.
        buffers.geometry_uploaded = true;

        Ok(GvfPrecomputeRaw {
            albnosh,
            blocking_distance,
            facesh,
        })
    }

    /// Upload cached geometry from a `GvfGeometryCache`. Used only when the
    /// geometry was computed on the CPU (GPU precompute disabled or fell back)
    /// but the per-timestep GVF still runs on the GPU. The GPU precompute path
    /// leaves geometry resident and does not call this.
    pub fn upload_geometry(&self, cache: &GvfGeometryCache) -> Result<(), String> {
        let num_azimuths = cache.azimuths.len();
        if num_azimuths == 0 {
            return Err("No azimuths in GVF geometry cache".to_string());
        }
        let (rows, cols) = (
            cache.azimuths[0].blocking_distance.nrows(),
            cache.azimuths[0].blocking_distance.ncols(),
        );
        let max_steps = cache.second as usize;

        let mut buf_cache = self.cached.lock().unwrap_or_else(|e| {
            eprintln!("WARNING: GVF GPU cache mutex was poisoned, recovering");
            let mut guard = e.into_inner();
            *guard = None;
            guard
        });

        self.ensure_buffers_locked(&mut buf_cache, rows, cols, num_azimuths, max_steps)?;
        let buffers = buf_cache
            .as_mut()
            .ok_or_else(|| "GVF GPU buffers missing after allocation".to_string())?;

        // Upload azimuth metadata
        let mut meta_data = Vec::with_capacity(num_azimuths);
        for (i, geom) in cache.azimuths.iter().enumerate() {
            meta_data.push(Self::azimuth_meta(geom.azimuth_deg, i, max_steps));
        }
        self.queue.write_buffer(
            &buffers.azimuth_meta_buffer,
            0,
            bytemuck::cast_slice(&meta_data),
        );

        // Upload shifts: flatten all azimuths' shifts into one buffer
        let total_shifts = num_azimuths * max_steps;
        let mut shift_data = vec![[0i32; 2]; total_shifts];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            for (n, &(dx, dy)) in geom.shifts.iter().enumerate() {
                shift_data[i * max_steps + n] = [dx as i32, dy as i32];
            }
        }
        self.queue
            .write_buffer(&buffers.shifts_buffer, 0, bytemuck::cast_slice(&shift_data));

        // Upload blocking_distance: [az × rows × cols] as u32
        let total_geom_pixels = num_azimuths * rows * cols;
        let mut bd_data = vec![0u32; total_geom_pixels];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            let offset = i * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    bd_data[offset + r * cols + c] = geom.blocking_distance[[r, c]] as u32;
                }
            }
        }
        self.queue.write_buffer(
            &buffers.blocking_distance_buffer,
            0,
            bytemuck::cast_slice(&bd_data),
        );

        // Upload facesh: [az × rows × cols] as f32
        let mut facesh_data = vec![0.0f32; total_geom_pixels];
        for (i, geom) in cache.azimuths.iter().enumerate() {
            let offset = i * rows * cols;
            if let Some(slice) = geom.facesh.as_slice() {
                facesh_data[offset..offset + rows * cols].copy_from_slice(slice);
            } else {
                for r in 0..rows {
                    for c in 0..cols {
                        facesh_data[offset + r * cols + c] = geom.facesh[[r, c]];
                    }
                }
            }
        }
        self.queue.write_buffer(
            &buffers.facesh_buffer,
            0,
            bytemuck::cast_slice(&facesh_data),
        );

        buffers.geometry_uploaded = true;
        Ok(())
    }

    /// Begin GPU dispatch for one timestep. Returns a pending token.
    pub fn dispatch_begin(
        &self,
        lup: ArrayView2<f32>,
        albshadow: ArrayView2<f32>,
        sunwall_mask: ArrayView2<f32>,
        first: f32,
        second: f32,
        lwall: f32,
        wall_albedo: f32,
    ) -> Result<GvfGpuPending, String> {
        let rows = lup.nrows();
        let cols = lup.ncols();
        let total_pixels = rows * cols;

        let mut cache = self.cached.lock().unwrap_or_else(|e| {
            eprintln!("WARNING: GVF GPU cache mutex was poisoned, recovering");
            let mut guard = e.into_inner();
            *guard = None;
            guard
        });

        let buffers = cache
            .as_mut()
            .ok_or_else(|| "GVF GPU buffers not allocated".to_string())?;

        if !buffers.geometry_uploaded {
            return Err("GVF geometry not uploaded — call upload_geometry() first".to_string());
        }
        if buffers.readback_inflight {
            return Err("GVF GPU readback already in flight".to_string());
        }
        if buffers.rows != rows || buffers.cols != cols {
            return Err(format!(
                "Grid size mismatch: buffers {}x{} vs input {}x{}",
                buffers.rows, buffers.cols, rows, cols
            ));
        }

        buffers.readback_inflight = true;

        // Upload uniform params
        let params = GvfParams {
            rows: rows as u32,
            cols: cols as u32,
            num_azimuths: buffers.num_azimuths as u32,
            max_steps: buffers.max_steps as u32,
            first,
            second,
            lwall,
            wall_albedo,
        };
        self.queue
            .write_buffer(&buffers.gvf_params_buffer, 0, bytemuck::bytes_of(&params));

        // Upload per-timestep inputs
        Self::write_2d_f32(&self.queue, &buffers.lup_buffer, &lup);
        Self::write_2d_f32(&self.queue, &buffers.albshadow_buffer, &albshadow);
        Self::write_2d_f32(&self.queue, &buffers.sunwall_mask_buffer, &sunwall_mask);

        // Dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GVF Cached Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GVF Cached Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group_0, &[]);
            pass.set_bind_group(1, &buffers.bind_group_1, &[]);
            pass.set_bind_group(2, &buffers.bind_group_2, &[]);

            let (wg_x, wg_y) = self.checked_workgroups_2d(rows, cols, 8, 8, "GVF cached")?;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy output to staging
        let output_bytes = (total_pixels * NUM_OUTPUT_CHANNELS * 4) as u64;
        encoder.copy_buffer_to_buffer(
            &buffers.outputs_buffer,
            0,
            &buffers.staging_buffer,
            0,
            output_bytes,
        );

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let buffer_slice = buffers.staging_buffer.slice(..output_bytes);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        Ok(GvfGpuPending {
            rows,
            cols,
            total_pixels,
            staging_size: output_bytes,
            submission_index,
            map_rx: receiver,
        })
    }

    /// Complete an in-flight dispatch and read back the 10 output arrays.
    pub fn dispatch_end(&self, pending: GvfGpuPending) -> Result<GvfGpuResult, String> {
        let result = (|| {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(pending.submission_index),
                    timeout: None,
                })
                .map_err(|e| format!("GPU poll failed: {:?}", e))?;

            pending
                .map_rx
                .recv()
                .map_err(|e| format!("Channel recv failed: {}", e))?
                .map_err(|e| format!("Failed to map staging buffer: {:?}", e))?;

            let cache = self.cached.lock().unwrap_or_else(|e| {
                eprintln!("WARNING: GVF GPU buffer cache mutex was poisoned, recovering");
                let mut guard = e.into_inner();
                *guard = None;
                guard
            });
            let buffers = cache
                .as_ref()
                .ok_or_else(|| "GVF GPU buffers missing".to_string())?;
            let buffer_slice = buffers.staging_buffer.slice(..pending.staging_size);
            let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
            let data = buffer_slice.get_mapped_range();
            let all_f32: &[f32] = bytemuck::cast_slice(&data);

            let n = pending.total_pixels;
            let extract = |ch: usize| -> Result<Array2<f32>, String> {
                Array2::from_shape_vec(
                    (pending.rows, pending.cols),
                    all_f32[ch * n..(ch + 1) * n].to_vec(),
                )
                .map_err(|e| format!("GVF output channel {}: {}", ch, e))
            };

            Ok(GvfGpuResult {
                lup: extract(0)?,
                alb: extract(1)?,
                lup_e: extract(2)?,
                alb_e: extract(3)?,
                lup_s: extract(4)?,
                alb_s: extract(5)?,
                lup_w: extract(6)?,
                alb_w: extract(7)?,
                lup_n: extract(8)?,
                alb_n: extract(9)?,
            })
        })();

        match self.cached.lock() {
            Ok(mut cache) => {
                if let Some(buffers) = cache.as_mut() {
                    buffers.readback_inflight = false;
                }
            }
            Err(e) => {
                eprintln!("WARNING: GVF GPU buffer cache mutex was poisoned, recovering");
                let mut cache = e.into_inner();
                *cache = None;
            }
        }

        result
    }

    // ── Buffer management ────────────────────────────────────────────────

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
        let geom_f32_bytes = (geom_pixels * 4) as u64;
        let geom_u32_bytes = (geom_pixels * 4) as u64;
        let albnosh_bytes = (NUM_ALBNOSH_CHANNELS * total_pixels * 4) as u64;
        let total_shifts = num_azimuths * max_steps;
        let shifts_bytes = (total_shifts * 8) as u64; // vec2<i32> = 8 bytes
        let meta_bytes = (num_azimuths * std::mem::size_of::<AzimuthMetaGpu>()) as u64;
        let output_bytes = (total_pixels * NUM_OUTPUT_CHANNELS * 4) as u64;

        // Capture OOM errors instead of panicking
        self.device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);

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
        // Geometry: written by the precompute shader (STORAGE) and read back
        // (COPY_SRC), read by the per-timestep shader (STORAGE), and also
        // uploaded from the CPU on the fallback path (COPY_DST).
        let geometry = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        // Uniforms
        let gvf_params_buffer = make(
            "GVF Params",
            std::mem::size_of::<GvfParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let precompute_params_buffer = make(
            "GVF Precompute Params",
            std::mem::size_of::<GvfPrecomputeParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Shared bind-group-0 inputs
        let azimuth_meta_buffer = make("GVF AzimuthMeta", meta_bytes, input);
        let shifts_buffer = make("GVF Shifts", shifts_bytes, input);

        // Shared geometry
        let blocking_distance_buffer = make("GVF BlockingDist", geom_u32_bytes, geometry);
        let facesh_buffer = make("GVF Facesh", geom_f32_bytes, geometry);

        // Precompute inputs + reduced output
        let buildings_buffer = make("GVF Precompute Buildings", pixel_bytes, input);
        let alb_buffer = make("GVF Precompute Albedo", pixel_bytes, input);
        let aspect_buffer = make("GVF Precompute Aspect", pixel_bytes, input);
        let wall_ht_buffer = make("GVF Precompute WallHt", pixel_bytes, input);
        let albnosh_buffer = make("GVF Precompute Albnosh", albnosh_bytes, output);

        // Per-timestep inputs + output
        let lup_buffer = make("GVF Lup", pixel_bytes, input);
        let albshadow_buffer = make("GVF Albshadow", pixel_bytes, input);
        let sunwall_mask_buffer = make("GVF SunwallMask", pixel_bytes, input);
        let outputs_buffer = make("GVF Outputs", output_bytes, output);

        // Staging
        let staging_buffer = make(
            "GVF Staging",
            output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let readback_staging_buffer = make(
            "GVF Precompute Readback Staging",
            albnosh_bytes + geom_u32_bytes + geom_f32_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Check for OOM before creating bind groups
        let err = pollster::block_on(self.device.pop_error_scope());
        if let Some(e) = err {
            *cache = None;
            return Err(format!(
                "GPU OOM allocating GVF buffers for {}x{} grid ({} azimuths): {}",
                rows, cols, num_azimuths, e
            ));
        }

        // Per-timestep bind groups
        let bind_group_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF BG0"),
            layout: &self.bg_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gvf_params_buffer.as_entire_binding(),
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
            label: Some("GVF BG1"),
            layout: &self.bg_layout_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blocking_distance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: facesh_buffer.as_entire_binding(),
                },
            ],
        });
        let bind_group_2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF BG2"),
            layout: &self.bg_layout_2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lup_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: albshadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sunwall_mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: outputs_buffer.as_entire_binding(),
                },
            ],
        });

        // Precompute bind groups
        let precompute_bind_group_0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG0"),
            layout: &self.bg_layout_p0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: precompute_params_buffer.as_entire_binding(),
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
        let precompute_bind_group_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG1"),
            layout: &self.bg_layout_p1,
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
        let precompute_bind_group_2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GVF Precompute BG2"),
            layout: &self.bg_layout_p2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: albnosh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blocking_distance_buffer.as_entire_binding(),
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
            gvf_params_buffer,
            precompute_params_buffer,
            azimuth_meta_buffer,
            shifts_buffer,
            blocking_distance_buffer,
            facesh_buffer,
            buildings_buffer,
            alb_buffer,
            aspect_buffer,
            wall_ht_buffer,
            albnosh_buffer,
            lup_buffer,
            albshadow_buffer,
            sunwall_mask_buffer,
            outputs_buffer,
            staging_buffer,
            readback_staging_buffer,
            albnosh_bytes,
            bd_bytes: geom_u32_bytes,
            facesh_bytes: geom_f32_bytes,
            bind_group_0,
            bind_group_1,
            bind_group_2,
            precompute_bind_group_0,
            precompute_bind_group_1,
            precompute_bind_group_2,
            geometry_uploaded: false,
            readback_inflight: false,
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

    /// Direction mask + shift offset + azimuth (rad) for one azimuth entry.
    fn azimuth_meta(azimuth_deg: f32, index: usize, max_steps: usize) -> AzimuthMetaGpu {
        let az = azimuth_deg;
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
        AzimuthMetaGpu {
            dir_mask,
            shift_offset: (index * max_steps) as u32,
            azimuth_rad: az * (std::f32::consts::PI / 180.0),
            _pad1: 0,
        }
    }

    fn write_azimuth_meta(
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        azimuths_deg: &[f32],
        max_steps: usize,
    ) {
        let meta_data: Vec<AzimuthMetaGpu> = azimuths_deg
            .iter()
            .enumerate()
            .map(|(i, &az)| Self::azimuth_meta(az, i, max_steps))
            .collect();
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&meta_data));
    }

    fn write_shifts(
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        shifts: &[Vec<(isize, isize)>],
        max_steps: usize,
    ) {
        let total_shifts = shifts.len() * max_steps;
        let mut shift_data = vec![[0i32; 2]; total_shifts];
        for (i, az_shifts) in shifts.iter().enumerate() {
            for (n, &(dx, dy)) in az_shifts.iter().enumerate() {
                shift_data[i * max_steps + n] = [dx as i32, dy as i32];
            }
        }
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&shift_data));
    }

    // ── Bind group layouts ───────────────────────────────────────────────

    fn bg0_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            // @binding(0) params: uniform
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
            // @binding(1) azimuth_meta: storage read
            Self::storage_ro_entry(1),
            // @binding(2) shifts: storage read
            Self::storage_ro_entry(2),
        ]
    }

    fn bg1_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            Self::storage_ro_entry(0), // blocking_distance
            Self::storage_ro_entry(1), // facesh
        ]
    }

    fn bg2_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![
            Self::storage_ro_entry(0), // lup
            Self::storage_ro_entry(1), // albshadow
            Self::storage_ro_entry(2), // sunwall_mask
            Self::storage_rw_entry(3), // outputs
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
