//! GPU-accelerated anisotropic sky radiation computation.
//!
//! Fuses the per-pixel × per-patch loop from `anisotropic_sky_pure()` onto a
//! single GPU compute dispatch. Each thread handles one pixel and iterates
//! over all sky patches, accumulating longwave and shortwave radiation.
//!
//! Shares `Arc<wgpu::Device>` and `Arc<wgpu::Queue>` with `ShadowGpuContext`.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};
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

// ── Uniform buffer (must match Params in anisotropic_sky.wgsl) ───────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AnisoParams {
    total_pixels: u32,
    cols: u32,
    rows: u32,
    n_patches: u32,
    n_pack: u32,
    cyl: u32,
    solar_altitude: f32,
    solar_azimuth: f32,
    ta: f32,
    albedo: f32,
    tgwall: f32,
    ewall: f32,
    rad_i: f32,
    rad_d: f32,
    psi: f32,
    rad_tot: f32,
}

// ── Cached GPU buffers ───────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
struct ViewSig3D {
    ptr: usize,
    d0: usize,
    d1: usize,
    d2: usize,
    s0: isize,
    s1: isize,
    s2: isize,
    sample_hash: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ViewSig2D {
    ptr: usize,
    d0: usize,
    d1: usize,
    s0: isize,
    s1: isize,
    sample_hash: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ViewSig1D {
    ptr: usize,
    d0: usize,
    s0: isize,
    sample_hash: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StaticInputSig {
    sh_ptr: usize,
    veg_ptr: usize,
    vb_ptr: usize,
    asvf_ptr: usize,
    valid_ptr: usize,
    patch_alt_ptr: usize,
    patch_azi_ptr: usize,
    ster_ptr: usize,
    rows: usize,
    cols: usize,
    n_pack: usize,
    n_patches: usize,
}

struct CachedBuffers {
    rows: usize,
    cols: usize,
    n_pack: usize,
    n_patches: usize,
    // Uniform
    params_buffer: wgpu::Buffer,
    // Per-pixel inputs
    shmat_buffer: wgpu::Buffer,
    vegshmat_buffer: wgpu::Buffer,
    vbshvegshmat_buffer: wgpu::Buffer,
    asvf_buffer: wgpu::Buffer,
    lup_buffer: wgpu::Buffer,
    valid_buffer: wgpu::Buffer,
    // Per-patch LUTs
    patch_alt_buffer: wgpu::Buffer,
    patch_azi_buffer: wgpu::Buffer,
    steradians_buffer: wgpu::Buffer,
    esky_band_buffer: wgpu::Buffer,
    lum_chi_buffer: wgpu::Buffer,
    // Outputs
    out_ldown_buffer: wgpu::Buffer,
    out_lside_buffer: wgpu::Buffer,
    out_kside_partial_buffer: wgpu::Buffer,
    out_drad_buffer: wgpu::Buffer,
    // Staging for readback
    staging_buffer: wgpu::Buffer,
    // Bind groups
    bind_group: wgpu::BindGroup,
    // Signatures for static inputs (skip uploads when unchanged)
    last_shmat_sig: Option<ViewSig3D>,
    last_vegshmat_sig: Option<ViewSig3D>,
    last_vbshmat_sig: Option<ViewSig3D>,
    last_asvf_sig: Option<ViewSig2D>,
    last_valid_sig: Option<ViewSig2D>,
    last_patch_alt_sig: Option<ViewSig1D>,
    last_patch_azi_sig: Option<ViewSig1D>,
    last_steradians_sig: Option<ViewSig1D>,
    last_static_input_sig: Option<StaticInputSig>,
    readback_inflight: bool,
}

// ── Public context ───────────────────────────────────────────────────────

pub struct AnisoGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    max_compute_workgroups_per_dimension: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    cached: Mutex<Option<CachedBuffers>>,
}

/// Result of GPU anisotropic sky computation.
pub struct AnisoGpuResult {
    pub ldown: Array2<f32>,
    pub lside: Array2<f32>,
    pub kside_partial: Array2<f32>,
    pub drad: Array2<f32>,
}

/// In-flight GPU dispatch token for overlapping CPU work with GPU readback.
pub struct AnisoGpuPending {
    rows: usize,
    cols: usize,
    total_pixels: usize,
    staging_size: u64,
    submission_index: wgpu::SubmissionIndex,
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl AnisoGpuContext {
    /// Create a new context, sharing device/queue from the shadow GPU context.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Anisotropic Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("anisotropic_sky.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Aniso Bind Group Layout"),
            entries: &Self::bind_group_layout_entries(),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Aniso Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Aniso Compute Pipeline"),
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
            bind_group_layout,
            cached: Mutex::new(None),
        }
    }

    /// Forget which static inputs are on the GPU, so the next dispatch uploads
    /// them all again.
    ///
    /// The per-array records compare an address and a few sampled values,
    /// which is sound only while the same arrays stay alive. The caller
    /// invokes this whenever it switches to another surface: a new surface's
    /// arrays can land on the addresses a freed surface used.
    pub fn invalidate_static_inputs(&self) {
        if let Ok(mut cache) = self.cached.lock() {
            if let Some(buffers) = cache.as_mut() {
                buffers.last_static_input_sig = None;
            }
        }
    }

    fn checked_workgroups_2d(
        &self,
        rows: usize,
        cols: usize,
        workgroup_x: u32,
        workgroup_y: u32,
        label: &str,
    ) -> Result<(u32, u32), String> {
        let workgroups_x = (cols as u32).div_ceil(workgroup_x);
        let workgroups_y = (rows as u32).div_ceil(workgroup_y);
        let limit = self.max_compute_workgroups_per_dimension;
        if workgroups_x > limit || workgroups_y > limit {
            return Err(format!(
                "{} dispatch exceeds GPU workgroup limit {}: got ({}, {}) for grid {}x{} and workgroup {}x{}",
                label, limit, workgroups_x, workgroups_y, rows, cols, workgroup_x, workgroup_y
            ));
        }
        Ok((workgroups_x, workgroups_y))
    }

    /// Begin anisotropic GPU dispatch and start asynchronous readback mapping.
    ///
    /// Call `dispatch_end()` to complete and retrieve arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_begin(
        &self,
        // Shadow matrices (bitpacked u8, shape rows×cols×n_pack)
        shmat: ArrayView3<u8>,
        vegshmat: ArrayView3<u8>,
        vbshvegshmat: ArrayView3<u8>,
        // Per-pixel arrays (shape rows×cols)
        asvf: ArrayView2<f32>,
        lup: ArrayView2<f32>,
        valid: ArrayView2<u8>,
        // Per-patch LUTs (length n_patches)
        patch_alt: ArrayView1<f32>,
        patch_azi: ArrayView1<f32>,
        steradians: ArrayView1<f32>,
        esky_band: ArrayView1<f32>,
        lum_chi: ArrayView1<f32>,
        // Scalar parameters
        solar_altitude: f32,
        solar_azimuth: f32,
        ta: f32,
        cyl: bool,
        albedo: f32,
        tgwall: f32,
        ewall: f32,
        rad_i: f32,
        rad_d: f32,
        psi: f32,
        rad_tot: f32,
    ) -> Result<AnisoGpuPending, String> {
        let rows = shmat.shape()[0];
        let cols = shmat.shape()[1];
        let n_pack = shmat.shape()[2];
        let n_patches = patch_alt.len();
        let total_pixels = rows * cols;

        let mut cache = self.cached.lock().unwrap_or_else(|e| {
            eprintln!("WARNING: anisotropic GPU buffer cache mutex was poisoned, recovering");
            let mut guard = e.into_inner();
            *guard = None; // discard stale state so buffers are recreated
            guard
        });
        if cache.as_ref().map(|b| b.readback_inflight).unwrap_or(false) {
            return Err("anisotropic GPU readback already in flight".to_string());
        }
        // Ensure buffers are allocated for this grid size while holding the cache lock,
        // so dimensions cannot change between the inflight check and allocation.
        self.ensure_buffers_locked(&mut cache, rows, cols, n_pack, n_patches);
        let buffers = cache
            .as_mut()
            .ok_or_else(|| "anisotropic GPU buffers missing after allocation".to_string())?;
        buffers.readback_inflight = true;

        // Static inputs are expected to stay stable during a run. If the backing
        // arrays change (new surface/tile), force re-upload of all cached static
        // buffers before signature checks below.
        let static_sig = StaticInputSig {
            sh_ptr: shmat.as_ptr() as usize,
            veg_ptr: vegshmat.as_ptr() as usize,
            vb_ptr: vbshvegshmat.as_ptr() as usize,
            asvf_ptr: asvf.as_ptr() as usize,
            valid_ptr: valid.as_ptr() as usize,
            patch_alt_ptr: patch_alt.as_ptr() as usize,
            patch_azi_ptr: patch_azi.as_ptr() as usize,
            ster_ptr: steradians.as_ptr() as usize,
            rows,
            cols,
            n_pack,
            n_patches,
        };
        if buffers.last_static_input_sig != Some(static_sig) {
            buffers.last_shmat_sig = None;
            buffers.last_vegshmat_sig = None;
            buffers.last_vbshmat_sig = None;
            buffers.last_asvf_sig = None;
            buffers.last_valid_sig = None;
            buffers.last_patch_alt_sig = None;
            buffers.last_patch_azi_sig = None;
            buffers.last_steradians_sig = None;
            buffers.last_static_input_sig = Some(static_sig);
        }

        // ── Upload uniform params ────────────────────────────────────────
        let params = AnisoParams {
            total_pixels: total_pixels as u32,
            cols: cols as u32,
            rows: rows as u32,
            n_patches: n_patches as u32,
            n_pack: n_pack as u32,
            cyl: if cyl { 1 } else { 0 },
            solar_altitude,
            solar_azimuth,
            ta,
            albedo,
            tgwall,
            ewall,
            rad_i,
            rad_d,
            psi,
            rad_tot,
        };
        self.queue
            .write_buffer(&buffers.params_buffer, 0, bytemuck::bytes_of(&params));

        // ── Upload shadow matrices ───────────────────────────────────────
        // Contiguous u8 data → reinterpreted as u32 on GPU (little-endian)
        let sh_sig = Self::sig_3d_u8(&shmat);
        if buffers.last_shmat_sig != Some(sh_sig) {
            Self::write_3d_u8_padded(&self.queue, &buffers.shmat_buffer, &shmat);
            buffers.last_shmat_sig = Some(sh_sig);
        }

        let veg_sig = Self::sig_3d_u8(&vegshmat);
        if buffers.last_vegshmat_sig != Some(veg_sig) {
            Self::write_3d_u8_padded(&self.queue, &buffers.vegshmat_buffer, &vegshmat);
            buffers.last_vegshmat_sig = Some(veg_sig);
        }

        let vb_sig = Self::sig_3d_u8(&vbshvegshmat);
        if buffers.last_vbshmat_sig != Some(vb_sig) {
            Self::write_3d_u8_padded(&self.queue, &buffers.vbshvegshmat_buffer, &vbshvegshmat);
            buffers.last_vbshmat_sig = Some(vb_sig);
        }

        // ── Upload per-pixel arrays ──────────────────────────────────────
        let asvf_sig = Self::sig_2d_f32(&asvf);
        if buffers.last_asvf_sig != Some(asvf_sig) {
            Self::write_2d_f32(&self.queue, &buffers.asvf_buffer, &asvf);
            buffers.last_asvf_sig = Some(asvf_sig);
        }

        Self::write_2d_f32(&self.queue, &buffers.lup_buffer, &lup);

        // Valid mask: u8 → pad to u32 alignment
        let valid_sig = Self::sig_2d_u8(&valid);
        if buffers.last_valid_sig != Some(valid_sig) {
            Self::write_2d_u8_padded(&self.queue, &buffers.valid_buffer, &valid);
            buffers.last_valid_sig = Some(valid_sig);
        }

        // ── Upload per-patch LUTs ────────────────────────────────────────
        let patch_alt_sig = Self::sig_1d_f32(&patch_alt);
        if buffers.last_patch_alt_sig != Some(patch_alt_sig) {
            Self::write_1d_f32(&self.queue, &buffers.patch_alt_buffer, &patch_alt);
            buffers.last_patch_alt_sig = Some(patch_alt_sig);
        }

        let patch_azi_sig = Self::sig_1d_f32(&patch_azi);
        if buffers.last_patch_azi_sig != Some(patch_azi_sig) {
            Self::write_1d_f32(&self.queue, &buffers.patch_azi_buffer, &patch_azi);
            buffers.last_patch_azi_sig = Some(patch_azi_sig);
        }

        let ster_sig = Self::sig_1d_f32(&steradians);
        if buffers.last_steradians_sig != Some(ster_sig) {
            Self::write_1d_f32(&self.queue, &buffers.steradians_buffer, &steradians);
            buffers.last_steradians_sig = Some(ster_sig);
        }

        Self::write_1d_f32(&self.queue, &buffers.esky_band_buffer, &esky_band);

        Self::write_1d_f32(&self.queue, &buffers.lum_chi_buffer, &lum_chi);

        // ── Dispatch compute shader ──────────────────────────────────────
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Aniso Sky Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Aniso Sky Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            let workgroup_x = 16u32;
            let workgroup_y = 16u32;
            let (workgroups_x, workgroups_y) = self.checked_workgroups_2d(
                rows,
                cols,
                workgroup_x,
                workgroup_y,
                "anisotropic sky",
            )?;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        // ── Copy outputs to staging buffer ───────────────────────────────
        let pixel_bytes = (total_pixels * 4) as u64; // f32 per pixel
        encoder.copy_buffer_to_buffer(
            &buffers.out_ldown_buffer,
            0,
            &buffers.staging_buffer,
            0,
            pixel_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.out_lside_buffer,
            0,
            &buffers.staging_buffer,
            pixel_bytes,
            pixel_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.out_kside_partial_buffer,
            0,
            &buffers.staging_buffer,
            pixel_bytes * 2,
            pixel_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.out_drad_buffer,
            0,
            &buffers.staging_buffer,
            pixel_bytes * 3,
            pixel_bytes,
        );
        // Wait only for this dispatch submission, not the whole device queue.
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let staging_size = pixel_bytes * 4;
        let buffer_slice = buffers.staging_buffer.slice(..staging_size);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        Ok(AnisoGpuPending {
            rows,
            cols,
            total_pixels,
            staging_size,
            submission_index,
            map_rx: receiver,
        })
    }

    /// Complete an in-flight anisotropic GPU dispatch and read back arrays.
    pub fn dispatch_end(&self, pending: AnisoGpuPending) -> Result<AnisoGpuResult, String> {
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
                eprintln!("WARNING: anisotropic GPU buffer cache mutex was poisoned, recovering");
                let mut guard = e.into_inner();
                *guard = None;
                guard
            });
            let buffers = cache
                .as_ref()
                .ok_or_else(|| "anisotropic GPU buffers missing".to_string())?;
            let buffer_slice = buffers.staging_buffer.slice(..pending.staging_size);
            let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
            let data = buffer_slice.get_mapped_range();
            let all_f32: &[f32] = bytemuck::cast_slice(&data);

            let ldown = Array2::from_shape_vec(
                (pending.rows, pending.cols),
                all_f32[..pending.total_pixels].to_vec(),
            )
            .map_err(|e| format!("ldown array: {}", e))?;

            let lside = Array2::from_shape_vec(
                (pending.rows, pending.cols),
                all_f32[pending.total_pixels..pending.total_pixels * 2].to_vec(),
            )
            .map_err(|e| format!("lside array: {}", e))?;

            let kside_partial = Array2::from_shape_vec(
                (pending.rows, pending.cols),
                all_f32[pending.total_pixels * 2..pending.total_pixels * 3].to_vec(),
            )
            .map_err(|e| format!("kside_partial array: {}", e))?;
            let drad = Array2::from_shape_vec(
                (pending.rows, pending.cols),
                all_f32[pending.total_pixels * 3..pending.total_pixels * 4].to_vec(),
            )
            .map_err(|e| format!("drad array: {}", e))?;

            Ok(AnisoGpuResult {
                ldown,
                lside,
                kside_partial,
                drad,
            })
        })();

        match self.cached.lock() {
            Ok(mut cache) => {
                if let Some(buffers) = cache.as_mut() {
                    buffers.readback_inflight = false;
                }
            }
            Err(e) => {
                eprintln!("WARNING: anisotropic GPU buffer cache mutex was poisoned, recovering");
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
        n_pack: usize,
        n_patches: usize,
    ) {
        // Reuse if dimensions match
        if let Some(ref c) = *cache {
            if c.rows == rows && c.cols == cols && c.n_pack == n_pack && c.n_patches == n_patches {
                return;
            }
        }

        let total_pixels = rows * cols;
        let pixel_bytes = (total_pixels * 4) as u64;
        let shadow_bytes = (total_pixels * n_pack) as u64;
        // Pad shadow buffer to u32 alignment
        let shadow_bytes_aligned = (shadow_bytes + 3) & !3;
        let patch_bytes = (n_patches * 4) as u64;
        // Valid mask: 1 byte per pixel, padded to u32
        let valid_bytes = ((total_pixels + 3) & !3) as u64;

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
            "Aniso Params",
            std::mem::size_of::<AnisoParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let shmat_buffer = make("Aniso shmat", shadow_bytes_aligned, input);
        let vegshmat_buffer = make("Aniso vegshmat", shadow_bytes_aligned, input);
        let vbshvegshmat_buffer = make("Aniso vbshvegshmat", shadow_bytes_aligned, input);
        let asvf_buffer = make("Aniso asvf", pixel_bytes, input);
        let lup_buffer = make("Aniso lup", pixel_bytes, input);
        let valid_buffer = make("Aniso valid", valid_bytes, input);

        let patch_alt_buffer = make("Aniso patch_alt", patch_bytes, input);
        let patch_azi_buffer = make("Aniso patch_azi", patch_bytes, input);
        let steradians_buffer = make("Aniso steradians", patch_bytes, input);
        let esky_band_buffer = make("Aniso esky_band", patch_bytes, input);
        let lum_chi_buffer = make("Aniso lum_chi", patch_bytes, input);

        let out_ldown_buffer = make("Aniso out_ldown", pixel_bytes, output);
        let out_lside_buffer = make("Aniso out_lside", pixel_bytes, output);
        let out_kside_partial_buffer = make("Aniso out_kside_partial", pixel_bytes, output);
        let out_drad_buffer = make("Aniso out_drad", pixel_bytes, output);

        let staging_buffer = make(
            "Aniso Staging",
            pixel_bytes * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Aniso Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shmat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vegshmat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vbshvegshmat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: asvf_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: lup_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: valid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: patch_alt_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: patch_azi_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: steradians_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: esky_band_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: lum_chi_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: out_ldown_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: out_lside_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: out_kside_partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: out_drad_buffer.as_entire_binding(),
                },
            ],
        });

        *cache = Some(CachedBuffers {
            rows,
            cols,
            n_pack,
            n_patches,
            params_buffer,
            shmat_buffer,
            vegshmat_buffer,
            vbshvegshmat_buffer,
            asvf_buffer,
            lup_buffer,
            valid_buffer,
            patch_alt_buffer,
            patch_azi_buffer,
            steradians_buffer,
            esky_band_buffer,
            lum_chi_buffer,
            out_ldown_buffer,
            out_lside_buffer,
            out_kside_partial_buffer,
            out_drad_buffer,
            staging_buffer,
            bind_group,
            last_shmat_sig: None,
            last_vegshmat_sig: None,
            last_vbshmat_sig: None,
            last_asvf_sig: None,
            last_valid_sig: None,
            last_patch_alt_sig: None,
            last_patch_azi_sig: None,
            last_steradians_sig: None,
            last_static_input_sig: None,
            readback_inflight: false,
        });
    }

    // ── Bind group layout ────────────────────────────────────────────────

    fn bind_group_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        vec![
            uniform_entry(0), // params
            storage_ro(1),    // shmat
            storage_ro(2),    // vegshmat
            storage_ro(3),    // vbshvegshmat
            storage_ro(4),    // asvf
            storage_ro(5),    // lup
            storage_ro(6),    // valid_mask
            storage_ro(7),    // patch_alt
            storage_ro(8),    // patch_azi
            storage_ro(9),    // steradians
            storage_ro(10),   // esky_band
            storage_ro(11),   // lum_chi
            storage_rw(12),   // out_ldown
            storage_rw(13),   // out_lside
            storage_rw(14),   // out_kside_partial
            storage_rw(15),   // out_drad
        ]
    }

    fn sig_3d_u8(arr: &ArrayView3<u8>) -> ViewSig3D {
        let sh = arr.shape();
        let st = arr.strides();
        let sample_hash = Self::sample_hash_3d_u8(arr);
        ViewSig3D {
            ptr: arr.as_ptr() as usize,
            d0: sh[0],
            d1: sh[1],
            d2: sh[2],
            s0: st[0],
            s1: st[1],
            s2: st[2],
            sample_hash,
        }
    }

    fn sig_2d_f32(arr: &ArrayView2<f32>) -> ViewSig2D {
        let sh = arr.shape();
        let st = arr.strides();
        let sample_hash = Self::sample_hash_2d_f32(arr);
        ViewSig2D {
            ptr: arr.as_ptr() as usize,
            d0: sh[0],
            d1: sh[1],
            s0: st[0],
            s1: st[1],
            sample_hash,
        }
    }

    fn sig_2d_u8(arr: &ArrayView2<u8>) -> ViewSig2D {
        let sh = arr.shape();
        let st = arr.strides();
        let sample_hash = Self::sample_hash_2d_u8(arr);
        ViewSig2D {
            ptr: arr.as_ptr() as usize,
            d0: sh[0],
            d1: sh[1],
            s0: st[0],
            s1: st[1],
            sample_hash,
        }
    }

    fn sig_1d_f32(arr: &ArrayView1<f32>) -> ViewSig1D {
        let sh = arr.shape();
        let st = arr.strides();
        let sample_hash = Self::sample_hash_1d_f32(arr);
        ViewSig1D {
            ptr: arr.as_ptr() as usize,
            d0: sh[0],
            s0: st[0],
            sample_hash,
        }
    }

    #[inline]
    fn hash_mix_u64(mut hash: u64, v: u64) -> u64 {
        // FNV-1a style mixing over 64-bit words.
        const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
        for b in v.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    fn sample_hash_3d_u8(arr: &ArrayView3<u8>) -> u64 {
        let sh = arr.shape();
        let r_last = sh[0] - 1;
        let c_last = sh[1] - 1;
        let k_last = sh[2] - 1;
        let coords = [
            (0, 0, 0),
            (r_last / 2, c_last / 2, k_last / 2),
            (r_last, c_last, k_last),
            (r_last / 4, c_last / 4, k_last / 4),
            ((3 * r_last) / 4, (3 * c_last) / 4, (3 * k_last) / 4),
            (r_last, 0, 0),
            (0, c_last, 0),
            (0, 0, k_last),
            (r_last, c_last, 0),
            (r_last, 0, k_last),
            (0, c_last, k_last),
        ];
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        hash = Self::hash_mix_u64(hash, sh[0] as u64);
        hash = Self::hash_mix_u64(hash, sh[1] as u64);
        hash = Self::hash_mix_u64(hash, sh[2] as u64);
        for &(r, c, k) in &coords {
            hash = Self::hash_mix_u64(hash, arr[[r, c, k]] as u64);
        }
        hash
    }

    fn sample_hash_2d_f32(arr: &ArrayView2<f32>) -> u64 {
        let sh = arr.shape();
        let r_last = sh[0] - 1;
        let c_last = sh[1] - 1;
        let coords = [
            (0, 0),
            (r_last / 2, c_last / 2),
            (r_last, c_last),
            (r_last / 4, c_last / 4),
            ((3 * r_last) / 4, (3 * c_last) / 4),
            (r_last, 0),
            (0, c_last),
        ];
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        hash = Self::hash_mix_u64(hash, sh[0] as u64);
        hash = Self::hash_mix_u64(hash, sh[1] as u64);
        for &(r, c) in &coords {
            hash = Self::hash_mix_u64(hash, arr[[r, c]].to_bits() as u64);
        }
        hash
    }

    fn sample_hash_2d_u8(arr: &ArrayView2<u8>) -> u64 {
        let sh = arr.shape();
        let r_last = sh[0] - 1;
        let c_last = sh[1] - 1;
        let coords = [
            (0, 0),
            (r_last / 2, c_last / 2),
            (r_last, c_last),
            (r_last / 4, c_last / 4),
            ((3 * r_last) / 4, (3 * c_last) / 4),
            (r_last, 0),
            (0, c_last),
        ];
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        hash = Self::hash_mix_u64(hash, sh[0] as u64);
        hash = Self::hash_mix_u64(hash, sh[1] as u64);
        for &(r, c) in &coords {
            hash = Self::hash_mix_u64(hash, arr[[r, c]] as u64);
        }
        hash
    }

    fn sample_hash_1d_f32(arr: &ArrayView1<f32>) -> u64 {
        let sh = arr.shape();
        let last = sh[0] - 1;
        let idx = [0, last / 4, last / 2, (3 * last) / 4, last];
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        hash = Self::hash_mix_u64(hash, sh[0] as u64);
        for &i in &idx {
            hash = Self::hash_mix_u64(hash, arr[i].to_bits() as u64);
        }
        hash
    }

    // ── Data conversion helpers ──────────────────────────────────────────

    #[inline]
    fn write_2d_f32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView2<f32>) {
        if let Some(slice) = arr.as_slice() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        } else {
            let packed = Self::contiguous_f32_2d(arr);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&packed));
        }
    }

    #[inline]
    fn write_1d_f32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView1<f32>) {
        if let Some(slice) = arr.as_slice() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        } else {
            let packed = Self::contiguous_f32_1d(arr);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&packed));
        }
    }

    #[inline]
    fn write_3d_u8_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView3<u8>) {
        if let Some(slice) = arr.as_slice() {
            if slice.len() % 4 == 0 {
                queue.write_buffer(buffer, 0, slice);
            } else {
                let mut packed = Vec::with_capacity((slice.len() + 3) & !3);
                packed.extend_from_slice(slice);
                while packed.len() % 4 != 0 {
                    packed.push(0);
                }
                queue.write_buffer(buffer, 0, &packed);
            }
        } else {
            let packed = Self::contiguous_bytes_3d(arr);
            queue.write_buffer(buffer, 0, &packed);
        }
    }

    #[inline]
    fn write_2d_u8_padded(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView2<u8>) {
        if let Some(slice) = arr.as_slice() {
            if slice.len() % 4 == 0 {
                queue.write_buffer(buffer, 0, slice);
            } else {
                let mut packed = Vec::with_capacity((slice.len() + 3) & !3);
                packed.extend_from_slice(slice);
                while packed.len() % 4 != 0 {
                    packed.push(0);
                }
                queue.write_buffer(buffer, 0, &packed);
            }
        } else {
            let packed = Self::contiguous_bytes_2d_u8(arr);
            queue.write_buffer(buffer, 0, &packed);
        }
    }

    /// Convert a 3D u8 ndarray to a contiguous byte Vec (row-major).
    fn contiguous_bytes_3d(arr: &ArrayView3<u8>) -> Vec<u8> {
        if let Some(slice) = arr.as_slice() {
            // Already contiguous — pad to u32 alignment
            let mut v = slice.to_vec();
            while v.len() % 4 != 0 {
                v.push(0);
            }
            v
        } else {
            let shape = arr.shape();
            let total = shape[0] * shape[1] * shape[2];
            let mut v = Vec::with_capacity((total + 3) & !3);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    for k in 0..shape[2] {
                        v.push(arr[[r, c, k]]);
                    }
                }
            }
            while v.len() % 4 != 0 {
                v.push(0);
            }
            v
        }
    }

    /// Convert a 2D f32 ndarray to a contiguous Vec<f32>.
    fn contiguous_f32_2d(arr: &ArrayView2<f32>) -> Vec<f32> {
        if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            let shape = arr.shape();
            let mut v = Vec::with_capacity(shape[0] * shape[1]);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    v.push(arr[[r, c]]);
                }
            }
            v
        }
    }

    /// Convert a 2D u8 ndarray to a contiguous byte Vec, padded to u32.
    fn contiguous_bytes_2d_u8(arr: &ArrayView2<u8>) -> Vec<u8> {
        let shape = arr.shape();
        let total = shape[0] * shape[1];
        let mut v = if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            let mut v = Vec::with_capacity(total);
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    v.push(arr[[r, c]]);
                }
            }
            v
        };
        while v.len() % 4 != 0 {
            v.push(0);
        }
        v
    }

    /// Convert a 1D f32 ndarray to a contiguous Vec<f32>.
    fn contiguous_f32_1d(arr: &ArrayView1<f32>) -> Vec<f32> {
        if let Some(slice) = arr.as_slice() {
            slice.to_vec()
        } else {
            arr.iter().cloned().collect()
        }
    }
}
