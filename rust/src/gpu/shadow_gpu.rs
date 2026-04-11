use ndarray::{Array2, Array3, ArrayView2};
use std::sync::Arc;

// ── Cross-platform GPU VRAM detection ────────────────────────────────────

/// Maximum `max_buffer_size` we trust from wgpu.  Values above this
/// are treated as bogus (DX12 reports 2^52, the D3D12 spec maximum).
const SANE_MAX_BUFFER_BYTES: u64 = 64 * 1024 * 1024 * 1024; // 64 GiB

/// Query actual dedicated GPU video memory via platform-specific APIs.
///
/// Returns `Some(bytes)` when the platform exposes real VRAM, `None`
/// otherwise (caller falls back to `max_buffer_size` heuristics).
#[cfg(target_os = "windows")]
fn query_platform_vram(info: &wgpu::AdapterInfo) -> Option<u64> {
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        // EnumAdapters1 returns Err (DXGI_ERROR_NOT_FOUND) when index
        // exceeds the adapter count, which terminates the loop via `?`.
        for i in 0u32..64 {
            let adapter = match factory.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => return None,
            };
            let desc = adapter.GetDesc1().ok()?;
            if desc.VendorId == info.vendor as u32
                && desc.DeviceId == info.device as u32
            {
                let vram = desc.DedicatedVideoMemory;
                if vram > 0 {
                    return Some(vram as u64);
                }
            }
        }
    }
    None
}

/// Linux/AMD: read `/sys/class/drm/card*/device/mem_info_vram_total` (amdgpu driver).
#[cfg(target_os = "linux")]
fn query_amdgpu_vram(info: &wgpu::AdapterInfo) -> Option<u64> {
    use std::fs;
    use std::path::Path;

    let drm = Path::new("/sys/class/drm");
    if !drm.is_dir() {
        return None;
    }
    for entry in fs::read_dir(drm).ok()? {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Skip renderD*, connector entries (card0-DP-1), etc.
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("card") || !name_str[4..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let dev_dir = entry.path().join("device");
        if !dev_dir.is_dir() {
            continue;
        }
        // Match vendor + device IDs (skip entries without these files)
        let vendor = match fs::read_to_string(dev_dir.join("vendor")) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let device = match fs::read_to_string(dev_dir.join("device")) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let vid = match u32::from_str_radix(vendor.trim().trim_start_matches("0x"), 16) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let did = match u32::from_str_radix(device.trim().trim_start_matches("0x"), 16) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if vid != info.vendor as u32 || did != info.device as u32 {
            continue;
        }
        let vram_path = dev_dir.join("mem_info_vram_total");
        if let Ok(content) = fs::read_to_string(&vram_path) {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                if bytes > 0 {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

/// Linux/NVIDIA: query dedicated VRAM via NVML (`libnvidia-ml.so.1`).
///
/// Uses `dlopen` to avoid a hard link-time dependency — returns `None`
/// if the NVIDIA driver is not installed or the library cannot be loaded.
#[cfg(target_os = "linux")]
fn query_nvml_vram(info: &wgpu::AdapterInfo) -> Option<u64> {
    // Only attempt for NVIDIA GPUs (vendor 0x10de)
    if info.vendor as u32 != 0x10de {
        return None;
    }

    use std::ffi::{c_char, c_uint, c_ulonglong, c_void};

    // NVML return type
    type NvmlReturn = c_uint;
    const NVML_SUCCESS: NvmlReturn = 0;

    // Opaque device handle
    type NvmlDevice = *mut c_void;

    // nvmlMemory_t — only the first field (total) is needed
    #[repr(C)]
    struct NvmlMemory {
        total: c_ulonglong,
        free: c_ulonglong,
        used: c_ulonglong,
    }

    // nvmlPciInfo_t (simplified — we only read busIdLegacy + pciDeviceId)
    #[repr(C)]
    struct NvmlPciInfo {
        bus_id_legacy: [c_char; 16],  // NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE
        domain: c_uint,
        bus: c_uint,
        device: c_uint,
        pci_device_id: c_uint,        // Combined device ID: (device_id << 16) | sub_vendor
        pci_sub_system_id: c_uint,
        bus_id: [c_char; 32],         // NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE
    }

    type InitFn = unsafe extern "C" fn() -> NvmlReturn;
    type ShutdownFn = unsafe extern "C" fn() -> NvmlReturn;
    type GetCountFn = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
    type GetHandleFn = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
    type GetMemoryFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;
    type GetPciFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlPciInfo) -> NvmlReturn;

    unsafe {
        let lib = libc::dlopen(
            b"libnvidia-ml.so.1\0".as_ptr() as *const c_char,
            libc::RTLD_LAZY,
        );
        if lib.is_null() {
            return None;
        }

        // RAII guard to ensure dlclose + nvmlShutdown
        struct NvmlGuard {
            lib: *mut c_void,
            shutdown: Option<ShutdownFn>,
        }
        impl Drop for NvmlGuard {
            fn drop(&mut self) {
                unsafe {
                    if let Some(f) = self.shutdown {
                        f();
                    }
                    libc::dlclose(self.lib);
                }
            }
        }

        macro_rules! sym {
            ($lib:expr, $name:literal, $ty:ty) => {{
                let ptr = libc::dlsym($lib, concat!($name, "\0").as_ptr() as *const c_char);
                if ptr.is_null() {
                    libc::dlclose($lib);
                    return None;
                }
                std::mem::transmute::<*mut c_void, $ty>(ptr)
            }};
        }

        let init: InitFn = sym!(lib, "nvmlInit_v2", InitFn);
        let shutdown: ShutdownFn = sym!(lib, "nvmlShutdown", ShutdownFn);
        let get_count: GetCountFn = sym!(lib, "nvmlDeviceGetCount_v2", GetCountFn);
        let get_handle: GetHandleFn = sym!(lib, "nvmlDeviceGetHandleByIndex_v2", GetHandleFn);
        let get_memory: GetMemoryFn = sym!(lib, "nvmlDeviceGetMemoryInfo", GetMemoryFn);
        let get_pci: GetPciFn = sym!(lib, "nvmlDeviceGetPciInfo_v3", GetPciFn);

        if init() != NVML_SUCCESS {
            libc::dlclose(lib);
            return None;
        }

        let mut guard = NvmlGuard {
            lib,
            shutdown: Some(shutdown),
        };

        let mut count: c_uint = 0;
        if get_count(&mut count) != NVML_SUCCESS || count == 0 {
            return None;
        }

        // Match adapter by PCI device ID from wgpu info
        let target_device_id = info.device as u32;

        for i in 0..count {
            let mut handle: NvmlDevice = std::ptr::null_mut();
            if get_handle(i, &mut handle) != NVML_SUCCESS {
                continue;
            }

            let mut pci = std::mem::zeroed::<NvmlPciInfo>();
            if get_pci(handle, &mut pci) != NVML_SUCCESS {
                continue;
            }

            // pci_device_id upper 16 bits = PCI device ID
            let nvml_dev_id = (pci.pci_device_id >> 16) & 0xFFFF;
            if nvml_dev_id != target_device_id {
                continue;
            }

            let mut mem = std::mem::zeroed::<NvmlMemory>();
            if get_memory(handle, &mut mem) != NVML_SUCCESS {
                continue;
            }

            if mem.total > 0 {
                // Prevent guard from leaking — drop happens automatically
                return Some(mem.total as u64);
            }
        }

        None
    }
}

/// Linux: try AMD sysfs first, then NVIDIA NVML.
#[cfg(target_os = "linux")]
fn query_platform_vram(info: &wgpu::AdapterInfo) -> Option<u64> {
    query_amdgpu_vram(info).or_else(|| query_nvml_vram(info))
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn query_platform_vram(_info: &wgpu::AdapterInfo) -> Option<u64> {
    None
}

/// Resolve the best available GPU memory budget.
///
/// Priority:
/// 1. Platform VRAM query (DXGI on Windows, sysfs on Linux)
/// 2. Metal `max_buffer_size` (reflects real unified memory)
/// 3. `max_buffer_size` if sane (< 64 GiB)
/// 4. None — Python-side will use a conservative fallback
fn resolve_gpu_memory_budget(
    backend: wgpu::Backend,
    info: &wgpu::AdapterInfo,
    max_buffer_size: u64,
) -> Option<u64> {
    // 1. Platform-specific query
    if let Some(vram) = query_platform_vram(info) {
        eprintln!(
            "[GPU] Detected {:.1} GiB dedicated video memory via platform API",
            vram as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        return Some(vram);
    }

    // 2. Metal: max_buffer_size reflects real unified memory
    if backend == wgpu::Backend::Metal {
        return Some(max_buffer_size);
    }

    // 3. Trust max_buffer_size only if it looks like a real VRAM value
    if max_buffer_size > 0 && max_buffer_size <= SANE_MAX_BUFFER_BYTES {
        return Some(max_buffer_size);
    }

    // 4. Bogus or zero — let Python use conservative default
    eprintln!(
        "[GPU] max_buffer_size={} exceeds sanity cap, cannot determine real VRAM",
        max_buffer_size
    );
    None
}

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

/// Result struct for GPU shadow calculations
pub struct GpuShadowResult {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Option<Array2<f32>>,
    pub veg_blocks_bldg_sh: Option<Array2<f32>>,
    pub propagated_veg_height: Option<Array2<f32>>,
    pub wall_sh: Option<Array2<f32>>,
    pub wall_sun: Option<Array2<f32>>,
    pub wall_sh_veg: Option<Array2<f32>>,
    pub face_sh: Option<Array2<f32>>,
    pub face_sun: Option<Array2<f32>>,
}

/// SVF-specific shadow result: only the 3 arrays needed for SVF computation.
/// Skips wall outputs entirely, reducing staging bandwidth by ~70%.
pub struct SvfShadowResult {
    pub bldg_sh: Array2<f32>,
    pub veg_sh: Option<Array2<f32>>,
    pub veg_blocks_bldg_sh: Option<Array2<f32>>,
}

/// Uniform buffer matching the SvfAccumParams struct in svf_accumulation.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SvfAccumParams {
    total_pixels: u32,
    cols: u32,
    rows: u32,
    weight_iso: f32,
    weight_n: f32,
    weight_e: f32,
    weight_s: f32,
    weight_w: f32,
    has_veg: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform buffer matching the U8PackParams struct in shadow_to_bitpack.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct U8PackParams {
    total_pixels: u32,
    cols: u32,
    rows: u32,
    n_pack: u32,
    matrix_words: u32,
    has_veg: u32,
    patch_byte_idx: u32,
    patch_bit_mask: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Bitpacked shadow matrices produced by GPU SVF path.
pub struct SvfBitpackedShadowResult {
    pub bldg_sh_matrix: Array3<u8>,
    pub veg_sh_matrix: Array3<u8>,
    pub veg_blocks_bldg_sh_matrix: Array3<u8>,
}

/// Result of GPU SVF accumulation — 15 arrays (5 building + 5 veg + 5 aveg).
pub struct SvfAccumResult {
    pub svf: Array2<f32>,
    pub svf_n: Array2<f32>,
    pub svf_e: Array2<f32>,
    pub svf_s: Array2<f32>,
    pub svf_w: Array2<f32>,
    pub svf_veg: Option<Array2<f32>>,
    pub svf_veg_n: Option<Array2<f32>>,
    pub svf_veg_e: Option<Array2<f32>>,
    pub svf_veg_s: Option<Array2<f32>>,
    pub svf_veg_w: Option<Array2<f32>>,
    pub svf_aveg: Option<Array2<f32>>,
    pub svf_aveg_n: Option<Array2<f32>>,
    pub svf_aveg_e: Option<Array2<f32>>,
    pub svf_aveg_s: Option<Array2<f32>>,
    pub svf_aveg_w: Option<Array2<f32>>,
}

/// Cached GPU buffers for shadow calculations.
/// Reused across calls when grid dimensions remain constant.
struct CachedBuffers {
    rows: usize,
    cols: usize,
    // Binding 0: Params (UNIFORM | COPY_DST)
    params_buffer: wgpu::Buffer,
    // Binding 1: DSM input (STORAGE | COPY_DST)
    dsm_buffer: wgpu::Buffer,
    // Binding 2: Building shadow output (STORAGE | COPY_SRC)
    bldg_shadow_buffer: wgpu::Buffer,
    // Binding 3: Propagated building height (STORAGE | COPY_SRC | COPY_DST)
    propagated_bldg_height_buffer: wgpu::Buffer,
    // Bindings 4-6: Vegetation inputs (STORAGE | COPY_DST)
    veg_canopy_buffer: wgpu::Buffer,
    veg_trunk_buffer: wgpu::Buffer,
    bush_buffer: wgpu::Buffer,
    // Bindings 7-9: Vegetation outputs (STORAGE | COPY_SRC)
    veg_shadow_buffer: wgpu::Buffer,
    propagated_veg_height_buffer: wgpu::Buffer,
    veg_blocks_bldg_shadow_buffer: wgpu::Buffer,
    // Bindings 10-11: Wall inputs (STORAGE | COPY_DST)
    walls_buffer: wgpu::Buffer,
    aspect_buffer: wgpu::Buffer,
    // Bindings 12-16: Wall outputs (STORAGE | COPY_SRC)
    wall_sh_buffer: wgpu::Buffer,
    wall_sun_buffer: wgpu::Buffer,
    wall_sh_veg_buffer: wgpu::Buffer,
    face_sh_buffer: wgpu::Buffer,
    face_sun_buffer: wgpu::Buffer,
    // Staging buffer for GPU -> CPU readback (MAP_READ | COPY_DST)
    staging_buffer: wgpu::Buffer,
    // Bind group (references all buffer handles)
    bind_group: wgpu::BindGroup,
    // --- SVF accumulation (populated by init_svf_accumulation) ---
    svf_params_buffer: Option<wgpu::Buffer>,
    svf_data_buffer: Option<wgpu::Buffer>,
    svf_result_staging: Option<wgpu::Buffer>,
    svf_bind_group: Option<wgpu::BindGroup>,
    svf_has_veg: bool,
    svf_num_arrays: usize, // 5 (no veg) or 15 (with veg)
    // --- Shadow bitpack accumulation (GPU-side across patches) ---
    shadow_u8_params_buffer: Option<wgpu::Buffer>,
    shadow_u8_output_buffer: Option<wgpu::Buffer>,
    shadow_u8_staging: Option<wgpu::Buffer>,
    shadow_u8_bind_group: Option<wgpu::BindGroup>,
    shadow_u8_packed_size: u64, // total bytes in packed output
    shadow_u8_n_pack: usize,
    shadow_u8_matrix_bytes: usize,
    shadow_u8_matrix_words: usize,
    shadow_u8_num_matrices: usize,
    // Signature of static inputs currently uploaded to GPU.
    last_static_input_sig: Option<StaticShadowInputSig>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StaticShadowInputSig {
    dsm_ptr: usize,
    veg_canopy_ptr: usize,
    veg_trunk_ptr: usize,
    bush_ptr: usize,
    walls_ptr: usize,
    aspect_ptr: usize,
    rows: usize,
    cols: usize,
    has_veg: bool,
    has_walls: bool,
}

/// GPU context for shadow calculations - maintains GPU resources across multiple calls
pub struct ShadowGpuContext {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    /// Adapter-reported maximum single buffer size in bytes.
    pub(crate) max_buffer_size: u64,
    /// Resolved GPU memory budget (real VRAM when detectable, else heuristic).
    pub(crate) gpu_memory_budget: Option<u64>,
    /// Adapter-reported maximum workgroups per dispatch dimension.
    max_compute_workgroups_per_dimension: u32,
    /// GPU backend (Metal, Vulkan, Dx12, Gl, etc.).
    pub(crate) backend: wgpu::Backend,
    pipeline: wgpu::ComputePipeline,
    wall_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    svf_pipeline: wgpu::ComputePipeline,
    svf_bind_group_layout: wgpu::BindGroupLayout,
    shadow_u8_pipeline: wgpu::ComputePipeline,
    shadow_u8_bind_group_layout: wgpu::BindGroupLayout,
    /// Cached buffers reused across calls with same grid dimensions
    cached: std::sync::Mutex<Option<CachedBuffers>>,
}

/// Uniform buffer struct for shadow shader parameters
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowParams {
    rows: u32,
    cols: u32,
    azimuth_rad: f32,
    altitude_rad: f32,
    sin_azimuth: f32,
    cos_azimuth: f32,
    tan_azimuth: f32,
    tan_altitude_by_scale: f32,
    scale: f32,
    max_index: f32,
    max_local_dsm_ht: f32,
    has_veg: u32,
    has_walls: u32,
    _padding: u32,
}

impl ShadowGpuContext {
    /// Initialize GPU context - call once at startup
    pub async fn new() -> Result<Self, String> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find suitable GPU adapter: {:?}", e))?;

        // Request higher limits for storage buffers and buffer sizes
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16; // We need 16 storage buffers
                                                          // Request native max buffer sizes for large SVF accumulation buffers
                                                          // (default 256 MiB is too small for packed 15-array SVF at 6.7M pixels)
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shadow Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow_propagation.wgsl").into()),
        });

        // Create bind group layout for all shadow types
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Bind Group Layout"),
            entries: &[
                // Binding 0: Params buffer (uniforms)
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
                // Binding 1: DSM input (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Building shadow output
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Propagated building height buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 4-9: Vegetation inputs and outputs
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 10-16: Wall inputs and outputs
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Propagation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let wall_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wall Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_wall_shadows"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- SVF accumulation pipeline ---
        let svf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SVF Accumulation Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("svf_accumulation.wgsl").into()),
        });

        let svf_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SVF Accumulation Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform params
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
                    // Binding 1: bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: veg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: veg_blocks_bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 4: svf_data (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let svf_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SVF Accumulation Pipeline Layout"),
            bind_group_layouts: &[&svf_bind_group_layout],
            push_constant_ranges: &[],
        });

        let svf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SVF Accumulation Pipeline"),
            layout: Some(&svf_pipeline_layout),
            module: &svf_shader,
            entry_point: Some("accumulate_svf"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Shadow bitpack accumulation pipeline ---
        let shadow_u8_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Bitpack Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow_to_bitpack.wgsl").into()),
        });

        let shadow_u8_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Bitpack Bind Group Layout"),
                entries: &[
                    // Binding 0: Uniform params
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
                    // Binding 1: bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: veg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 3: veg_blocks_bldg_sh (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 4: packed_output bit matrices (read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shadow_u8_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Bitpack Pipeline Layout"),
                bind_group_layouts: &[&shadow_u8_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shadow_u8_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shadow Bitpack Pipeline"),
            layout: Some(&shadow_u8_pipeline_layout),
            module: &shadow_u8_shader,
            entry_point: Some("shadow_to_bitpack"),
            compilation_options: Default::default(),
            cache: None,
        });

        let info = adapter.get_info();
        let backend = info.backend;
        let gpu_memory_budget =
            resolve_gpu_memory_budget(backend, &info, adapter_limits.max_buffer_size);

        Ok(Self {
            device,
            queue,
            max_buffer_size: adapter_limits.max_buffer_size,
            gpu_memory_budget,
            max_compute_workgroups_per_dimension: adapter_limits
                .max_compute_workgroups_per_dimension,
            backend,
            pipeline,
            wall_pipeline,
            bind_group_layout,
            svf_pipeline,
            svf_bind_group_layout,
            shadow_u8_pipeline,
            shadow_u8_bind_group_layout,
            cached: std::sync::Mutex::new(None),
        })
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

    /// Allocate a fresh set of GPU buffers for the given grid dimensions.
    fn allocate_buffers(&self, rows: usize, cols: usize) -> CachedBuffers {
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let params_size = std::mem::size_of::<ShadowParams>() as u64;

        // Helper to create a storage buffer with given usage flags
        let make_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let input_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let output_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let params_buffer = make_buffer(
            "Shadow Params Buffer",
            params_size,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let dsm_buffer = make_buffer("DSM Buffer", buffer_size, input_usage);
        let bldg_shadow_buffer = make_buffer("Building Shadow Buffer", buffer_size, output_usage);
        let propagated_bldg_height_buffer = make_buffer(
            "Propagated Building Height Buffer",
            buffer_size,
            working_usage,
        );
        let veg_canopy_buffer = make_buffer("Veg Canopy Buffer", buffer_size, input_usage);
        let veg_trunk_buffer = make_buffer("Veg Trunk Buffer", buffer_size, input_usage);
        let bush_buffer = make_buffer("Bush Buffer", buffer_size, input_usage);
        let veg_shadow_buffer = make_buffer("Veg Shadow Buffer", buffer_size, output_usage);
        let propagated_veg_height_buffer =
            make_buffer("Propagated Veg Height Buffer", buffer_size, working_usage);
        let veg_blocks_bldg_shadow_buffer =
            make_buffer("Veg Blocks Bldg Shadow Buffer", buffer_size, output_usage);
        let walls_buffer = make_buffer("Walls Buffer", buffer_size, input_usage);
        let aspect_buffer = make_buffer("Aspect Buffer", buffer_size, input_usage);
        let wall_sh_buffer = make_buffer("Wall Shadow Buffer", buffer_size, output_usage);
        let wall_sun_buffer = make_buffer("Wall Sun Buffer", buffer_size, output_usage);
        let wall_sh_veg_buffer = make_buffer("Wall Shadow Veg Buffer", buffer_size, output_usage);
        let face_sh_buffer = make_buffer("Face Shadow Buffer", buffer_size, output_usage);
        let face_sun_buffer = make_buffer("Face Sun Buffer", buffer_size, output_usage);

        let staging_buffer = make_buffer(
            "Staging Buffer",
            buffer_size * 10,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dsm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: propagated_bldg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: veg_canopy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: veg_trunk_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bush_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: propagated_veg_height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: walls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: aspect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wall_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wall_sun_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wall_sh_veg_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: face_sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: face_sun_buffer.as_entire_binding(),
                },
            ],
        });


        CachedBuffers {
            rows,
            cols,
            params_buffer,
            dsm_buffer,
            bldg_shadow_buffer,
            propagated_bldg_height_buffer,
            veg_canopy_buffer,
            veg_trunk_buffer,
            bush_buffer,
            veg_shadow_buffer,
            propagated_veg_height_buffer,
            veg_blocks_bldg_shadow_buffer,
            walls_buffer,
            aspect_buffer,
            wall_sh_buffer,
            wall_sun_buffer,
            wall_sh_veg_buffer,
            face_sh_buffer,
            face_sun_buffer,
            staging_buffer,
            bind_group,
            svf_params_buffer: None,
            svf_data_buffer: None,
            svf_result_staging: None,
            svf_bind_group: None,
            svf_has_veg: false,
            svf_num_arrays: 0,
            shadow_u8_params_buffer: None,
            shadow_u8_output_buffer: None,
            shadow_u8_staging: None,
            shadow_u8_bind_group: None,
            shadow_u8_packed_size: 0,
            shadow_u8_n_pack: 0,
            shadow_u8_matrix_bytes: 0,
            shadow_u8_matrix_words: 0,
            shadow_u8_num_matrices: 0,
            last_static_input_sig: None,
        }
    }

    /// Optimized version accepting ArrayView to avoid unnecessary copies
    #[allow(clippy::too_many_arguments)]
    pub fn compute_all_shadows_view(
        &self,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
        walls_opt: Option<ArrayView2<f32>>,
        aspect_opt: Option<ArrayView2<f32>>,
        need_propagated_veg_height: bool,
        need_full_wall_outputs: bool,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
        max_shadow_distance_m: f32,
    ) -> Result<GpuShadowResult, String> {
        let (rows, cols) = dsm.dim();
        let total_pixels = rows * cols;

        // Check if vegetation inputs are provided
        let has_veg =
            veg_canopy_dsm_opt.is_some() && veg_trunk_dsm_opt.is_some() && bush_opt.is_some();
        let has_walls = walls_opt.is_some() && aspect_opt.is_some();

        // Precompute trigonometric values
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let sin_azimuth = azimuth_rad.sin();
        let cos_azimuth = azimuth_rad.cos();
        let tan_azimuth = azimuth_rad.tan();
        // See shadowing.rs scale note: solweig convention is `scale = pixel_size_m`,
        // so the vertical-drop factor multiplies rather than divides.
        let tan_altitude_by_scale = altitude_rad.tan() * scale;
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let height_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_reach_m = if max_shadow_distance_m > 0.0 { height_reach_m.min(max_shadow_distance_m) } else { height_reach_m };
        let max_index = (max_reach_m / scale).ceil();

        let params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth,
            cos_azimuth,
            tan_azimuth,
            tan_altitude_by_scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: if has_walls { 1 } else { 0 },
            _padding: 0,
        };

        // Get or create cached buffers for this grid size
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Dynamic params change every timestep.
        self.queue
            .write_buffer(&buffers.params_buffer, 0, bytemuck::cast_slice(&[params]));

        // Static inputs are invariant across timesteps for a tile; avoid
        // re-uploading when backing arrays are unchanged.
        let static_sig = StaticShadowInputSig {
            dsm_ptr: dsm.as_ptr() as usize,
            veg_canopy_ptr: veg_canopy_dsm_opt.map_or(0, |a| a.as_ptr() as usize),
            veg_trunk_ptr: veg_trunk_dsm_opt.map_or(0, |a| a.as_ptr() as usize),
            bush_ptr: bush_opt.map_or(0, |a| a.as_ptr() as usize),
            walls_ptr: walls_opt.map_or(0, |a| a.as_ptr() as usize),
            aspect_ptr: aspect_opt.map_or(0, |a| a.as_ptr() as usize),
            rows,
            cols,
            has_veg,
            has_walls,
        };

        if buffers.last_static_input_sig != Some(static_sig) {
            Self::write_2d_f32(&self.queue, &buffers.dsm_buffer, &dsm);
            if has_veg {
                let veg_canopy = veg_canopy_dsm_opt
                    .ok_or_else(|| "Vegetation canopy missing despite has_veg=true".to_string())?;
                let veg_trunk = veg_trunk_dsm_opt
                    .ok_or_else(|| "Vegetation trunk missing despite has_veg=true".to_string())?;
                let bush = bush_opt
                    .ok_or_else(|| "Bush raster missing despite has_veg=true".to_string())?;
                Self::write_2d_f32(&self.queue, &buffers.veg_canopy_buffer, &veg_canopy);
                Self::write_2d_f32(&self.queue, &buffers.veg_trunk_buffer, &veg_trunk);
                Self::write_2d_f32(&self.queue, &buffers.bush_buffer, &bush);
            }
            if has_walls {
                let walls =
                    walls_opt.ok_or_else(|| "Walls missing despite has_walls=true".to_string())?;
                let aspect = aspect_opt
                    .ok_or_else(|| "Aspect missing despite has_walls=true".to_string())?;
                Self::write_2d_f32(&self.queue, &buffers.walls_buffer, &walls);
                Self::write_2d_f32(&self.queue, &buffers.aspect_buffer, &aspect);
            }
            buffers.last_static_input_sig = Some(static_sig);
        }

        // Encode and submit compute passes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Shadow Compute Encoder"),
            });

        // Reset mutable propagation buffers from static input buffers on-GPU.
        encoder.copy_buffer_to_buffer(
            &buffers.dsm_buffer,
            0,
            &buffers.propagated_bldg_height_buffer,
            0,
            buffer_size,
        );
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_canopy_buffer,
                0,
                &buffers.propagated_veg_height_buffer,
                0,
                buffer_size,
            );
        }

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let (num_workgroups_x, num_workgroups_y) = self.checked_workgroups_2d(
            rows,
            cols,
            workgroup_size_x,
            workgroup_size_y,
            "shadow propagation",
        )?;

        // First pass: Main shadow propagation
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Second pass: Wall shadows (if enabled)
        if has_walls {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wall Shadow Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.wall_pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy only required outputs to staging to reduce readback bandwidth.
        let include_prop_veg = has_veg && need_propagated_veg_height;
        let mut write_offset = 0u64;
        encoder.copy_buffer_to_buffer(
            &buffers.bldg_shadow_buffer,
            0,
            &buffers.staging_buffer,
            write_offset,
            buffer_size,
        );
        write_offset += buffer_size;

        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                write_offset,
                buffer_size,
            );
            write_offset += buffer_size;
            encoder.copy_buffer_to_buffer(
                &buffers.veg_blocks_bldg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                write_offset,
                buffer_size,
            );
            write_offset += buffer_size;
            if include_prop_veg {
                encoder.copy_buffer_to_buffer(
                    &buffers.propagated_veg_height_buffer,
                    0,
                    &buffers.staging_buffer,
                    write_offset,
                    buffer_size,
                );
                write_offset += buffer_size;
            }
        }

        if has_walls {
            if need_full_wall_outputs {
                encoder.copy_buffer_to_buffer(
                    &buffers.wall_sh_buffer,
                    0,
                    &buffers.staging_buffer,
                    write_offset,
                    buffer_size,
                );
                write_offset += buffer_size;
            }
            encoder.copy_buffer_to_buffer(
                &buffers.wall_sun_buffer,
                0,
                &buffers.staging_buffer,
                write_offset,
                buffer_size,
            );
            write_offset += buffer_size;
            if need_full_wall_outputs {
                encoder.copy_buffer_to_buffer(
                    &buffers.wall_sh_veg_buffer,
                    0,
                    &buffers.staging_buffer,
                    write_offset,
                    buffer_size,
                );
                write_offset += buffer_size;
                encoder.copy_buffer_to_buffer(
                    &buffers.face_sh_buffer,
                    0,
                    &buffers.staging_buffer,
                    write_offset,
                    buffer_size,
                );
                write_offset += buffer_size;
                encoder.copy_buffer_to_buffer(
                    &buffers.face_sun_buffer,
                    0,
                    &buffers.staging_buffer,
                    write_offset,
                    buffer_size,
                );
                write_offset += buffer_size;
            }
        }
        let read_size = write_offset;

        let submission_index = self.queue.submit(Some(encoder.finish()));

        // Read back only populated bytes from staging buffer.
        let buffer_slice = buffers.staging_buffer.slice(..read_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| format!("GPU poll failed while reading shadow buffers: {:?}", e))?;
        receiver
            .recv()
            .map_err(|e| format!("Failed waiting for shadow buffer mapping: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
        let data = buffer_slice.get_mapped_range();
        let all_data: &[f32] = bytemuck::cast_slice(&data);

        // Extract building shadow
        let mut read_offset_px = 0usize;
        let bldg_sh = Array2::from_shape_vec(
            (rows, cols),
            all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
        )
        .map_err(|e| format!("Failed to create building shadow array: {}", e))?;
        read_offset_px += total_pixels;

        // Extract vegetation results if enabled
        let (veg_sh, veg_blocks_bldg_sh, propagated_veg_height) = if has_veg {
            let veg_sh = Array2::from_shape_vec(
                (rows, cols),
                all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
            )
            .map_err(|e| format!("Failed to create vegetation shadow array: {}", e))?;
            read_offset_px += total_pixels;
            let veg_blocks = Array2::from_shape_vec(
                (rows, cols),
                all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
            )
            .map_err(|e| format!("Failed to create vegetation-blocking shadow array: {}", e))?;
            read_offset_px += total_pixels;
            let prop_veg = if include_prop_veg {
                let arr = Array2::from_shape_vec(
                    (rows, cols),
                    all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
                )
                .map_err(|e| {
                    format!("Failed to create propagated vegetation height array: {}", e)
                })?;
                read_offset_px += total_pixels;
                Some(arr)
            } else {
                None
            };
            (Some(veg_sh), Some(veg_blocks), prop_veg)
        } else {
            (None, None, None)
        };

        // Extract wall results if enabled
        let (wall_sh, wall_sun, wall_sh_veg, face_sh, face_sun) = if has_walls {
            let wall_sh = if need_full_wall_outputs {
                let arr = Array2::from_shape_vec(
                    (rows, cols),
                    all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
                )
                .map_err(|e| format!("Failed to create wall shadow array: {}", e))?;
                read_offset_px += total_pixels;
                Some(arr)
            } else {
                None
            };
            let wall_sun = Array2::from_shape_vec(
                (rows, cols),
                all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
            )
            .map_err(|e| format!("Failed to create wall sunlit array: {}", e))?;
            read_offset_px += total_pixels;
            if need_full_wall_outputs {
                let wall_sh_veg = Array2::from_shape_vec(
                    (rows, cols),
                    all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
                )
                .map_err(|e| format!("Failed to create wall vegetation-shadow array: {}", e))?;
                read_offset_px += total_pixels;
                let face_sh = Array2::from_shape_vec(
                    (rows, cols),
                    all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
                )
                .map_err(|e| format!("Failed to create wall face-shadow array: {}", e))?;
                read_offset_px += total_pixels;
                let face_sun = Array2::from_shape_vec(
                    (rows, cols),
                    all_data[read_offset_px..read_offset_px + total_pixels].to_vec(),
                )
                .map_err(|e| format!("Failed to create wall face-sun array: {}", e))?;
                (
                    wall_sh,
                    Some(wall_sun),
                    Some(wall_sh_veg),
                    Some(face_sh),
                    Some(face_sun),
                )
            } else {
                (wall_sh, Some(wall_sun), None, None, None)
            }
        } else {
            (None, None, None, None, None)
        };

        Ok(GpuShadowResult {
            bldg_sh,
            veg_sh,
            veg_blocks_bldg_sh,
            propagated_veg_height,
            wall_sh,
            wall_sun,
            wall_sh_veg,
            face_sh,
            face_sun,
        })
    }

    /// SVF-optimized shadow computation.
    ///
    /// Compared to `compute_all_shadows_view()`, this:
    /// - Skips the wall shader dispatch entirely
    /// - Skips writing wall/aspect input buffers (saves ~50MB/call)
    /// - Copies only 3 arrays to staging instead of 10 (~70% less readback)
    #[allow(clippy::too_many_arguments)]
    pub fn compute_shadows_for_svf(
        &self,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
        max_shadow_distance_m: f32,
    ) -> Result<SvfShadowResult, String> {
        let (rows, cols) = dsm.dim();
        let total_pixels = rows * cols;

        // Handle zenith case (altitude >= 89.5°): no shadows from directly overhead
        if altitude_deg >= 89.5 {
            let dim = (rows, cols);
            return Ok(SvfShadowResult {
                bldg_sh: Array2::ones(dim),
                veg_sh: if veg_canopy_dsm_opt.is_some() {
                    Some(Array2::ones(dim))
                } else {
                    None
                },
                veg_blocks_bldg_sh: if veg_canopy_dsm_opt.is_some() {
                    Some(Array2::ones(dim))
                } else {
                    None
                },
            });
        }

        let has_veg =
            veg_canopy_dsm_opt.is_some() && veg_trunk_dsm_opt.is_some() && bush_opt.is_some();

        let get_slice = |view: ArrayView2<f32>| -> Vec<f32> {
            if let Some(slice) = view.as_slice() {
                slice.to_vec()
            } else {
                view.iter().copied().collect()
            }
        };

        let dsm_data = get_slice(dsm);

        // Precompute trigonometric values
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let height_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_reach_m = if max_shadow_distance_m > 0.0 { height_reach_m.min(max_shadow_distance_m) } else { height_reach_m };
        let max_index = (max_reach_m / scale).ceil();

        let params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth: azimuth_rad.sin(),
            cos_azimuth: azimuth_rad.cos(),
            tan_azimuth: azimuth_rad.tan(),
            // Guard: f32 tan(~90°) can return a large negative number when
            // the f32 representation of π/2 slightly exceeds the true value.
            // For above-horizon patches, tan must be non-negative; use abs()
            // so the large negative becomes large positive → dz exceeds
            // max_local_dsm_ht immediately → no shadow (physically correct).
            //
            // See shadowing.rs scale note: solweig convention is
            // `scale = pixel_size_m`, so the vertical-drop factor multiplies.
            tan_altitude_by_scale: altitude_rad.tan().abs() * scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: 0, // SVF never uses walls
            _padding: 0,
        };

        // Get or create cached buffers
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_ref()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Write only needed inputs (skip walls/aspect entirely)
        self.queue
            .write_buffer(&buffers.params_buffer, 0, bytemuck::cast_slice(&[params]));
        self.queue
            .write_buffer(&buffers.dsm_buffer, 0, bytemuck::cast_slice(&dsm_data));
        self.queue.write_buffer(
            &buffers.propagated_bldg_height_buffer,
            0,
            bytemuck::cast_slice(&dsm_data),
        );

        if has_veg {
            let veg_canopy = veg_canopy_dsm_opt
                .ok_or_else(|| "Vegetation canopy DSM missing despite has_veg=true".to_string())?;
            let veg_trunk = veg_trunk_dsm_opt
                .ok_or_else(|| "Vegetation trunk DSM missing despite has_veg=true".to_string())?;
            let bush =
                bush_opt.ok_or_else(|| "Bush raster missing despite has_veg=true".to_string())?;
            let veg_canopy_data = get_slice(veg_canopy);
            let veg_trunk_data = get_slice(veg_trunk);
            let bush_data = get_slice(bush);
            self.queue.write_buffer(
                &buffers.veg_canopy_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
            self.queue.write_buffer(
                &buffers.veg_trunk_buffer,
                0,
                bytemuck::cast_slice(&veg_trunk_data),
            );
            self.queue
                .write_buffer(&buffers.bush_buffer, 0, bytemuck::cast_slice(&bush_data));
            self.queue.write_buffer(
                &buffers.propagated_veg_height_buffer,
                0,
                bytemuck::cast_slice(&veg_canopy_data),
            );
        }

        // Encode: shadow propagation only (no wall pass)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Shadow Compute Encoder"),
            });

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let (num_workgroups_x, num_workgroups_y) = self.checked_workgroups_2d(
            rows,
            cols,
            workgroup_size_x,
            workgroup_size_y,
            "svf shadow propagation",
        )?;

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SVF Shadow Propagation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &buffers.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }
        // No wall pass — SVF never uses walls

        // Copy only 3 arrays to staging (instead of up to 10)
        encoder.copy_buffer_to_buffer(
            &buffers.bldg_shadow_buffer,
            0,
            &buffers.staging_buffer,
            0,
            buffer_size,
        );
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                buffer_size,
                buffer_size,
            );
            encoder.copy_buffer_to_buffer(
                &buffers.veg_blocks_bldg_shadow_buffer,
                0,
                &buffers.staging_buffer,
                buffer_size * 2,
                buffer_size,
            );
        }

        let submission_index = self.queue.submit(Some(encoder.finish()));

        // Map only what we need (1 or 3 arrays)
        let read_size = if has_veg {
            buffer_size * 3
        } else {
            buffer_size
        };
        let buffer_slice = buffers.staging_buffer.slice(..read_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| format!("GPU poll failed while reading SVF shadow buffers: {:?}", e))?;
        receiver
            .recv()
            .map_err(|e| format!("Failed waiting for SVF shadow buffer mapping: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let _unmap_guard = MappedBufferGuard::new(&buffers.staging_buffer);
        let data = buffer_slice.get_mapped_range();
        let all_data: &[f32] = bytemuck::cast_slice(&data);

        let bldg_sh = Array2::from_shape_vec((rows, cols), all_data[..total_pixels].to_vec())
            .map_err(|e| format!("Failed to create bldg_sh array: {}", e))?;

        let (veg_sh, veg_blocks_bldg_sh) = if has_veg {
            let veg = Array2::from_shape_vec(
                (rows, cols),
                all_data[total_pixels..total_pixels * 2].to_vec(),
            )
            .map_err(|e| format!("Failed to create SVF vegetation shadow array: {}", e))?;
            let veg_blocks = Array2::from_shape_vec(
                (rows, cols),
                all_data[total_pixels * 2..total_pixels * 3].to_vec(),
            )
            .map_err(|e| {
                format!(
                    "Failed to create SVF vegetation-blocking shadow array: {}",
                    e
                )
            })?;
            (Some(veg), Some(veg_blocks))
        } else {
            (None, None)
        };

        Ok(SvfShadowResult {
            bldg_sh,
            veg_sh,
            veg_blocks_bldg_sh,
        })
    }

    /// Initialize SVF accumulation buffers. Call once before the 153-patch loop.
    ///
    /// Allocates the packed SVF data buffer (15 × pixels for veg, 5 × for no-veg),
    /// zeroes it, and creates the bind group referencing shadow output buffers.
    /// Also writes static inputs (DSM, veg) to shadow buffers once.
    pub fn init_svf_accumulation(
        &self,
        rows: usize,
        cols: usize,
        has_veg: bool,
        total_patches: usize,
        dsm: ArrayView2<f32>,
        veg_canopy_dsm_opt: Option<ArrayView2<f32>>,
        veg_trunk_dsm_opt: Option<ArrayView2<f32>>,
        bush_opt: Option<ArrayView2<f32>>,
    ) -> Result<(), String> {
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let num_arrays: usize = if has_veg { 15 } else { 5 };
        let svf_data_size = buffer_size * num_arrays as u64;

        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        // Ensure shadow buffers are allocated
        let needs_realloc = match cache_guard.as_ref() {
            Some(c) => c.rows != rows || c.cols != cols,
            None => true,
        };
        if needs_realloc {
            *cache_guard = Some(self.allocate_buffers(rows, cols));
        }

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache unexpectedly empty".to_string())?;

        // Write static inputs to shadow buffers once (avoids re-uploading per patch)
        let get_slice = |view: ArrayView2<f32>| -> Vec<f32> {
            if let Some(slice) = view.as_slice() {
                slice.to_vec()
            } else {
                view.iter().copied().collect()
            }
        };

        let dsm_data = get_slice(dsm);
        self.queue
            .write_buffer(&buffers.dsm_buffer, 0, bytemuck::cast_slice(&dsm_data));

        if has_veg {
            if let (Some(vc), Some(vt), Some(b)) = (veg_canopy_dsm_opt, veg_trunk_dsm_opt, bush_opt)
            {
                let vc_data = get_slice(vc);
                let vt_data = get_slice(vt);
                let b_data = get_slice(b);
                self.queue.write_buffer(
                    &buffers.veg_canopy_buffer,
                    0,
                    bytemuck::cast_slice(&vc_data),
                );
                self.queue.write_buffer(
                    &buffers.veg_trunk_buffer,
                    0,
                    bytemuck::cast_slice(&vt_data),
                );
                self.queue
                    .write_buffer(&buffers.bush_buffer, 0, bytemuck::cast_slice(&b_data));
            }
        }

        // Reuse SVF accumulation buffers when dimensions and veg config match.
        // This avoids GPU buffer allocation overhead in tiled mode where all tiles
        // have the same dimensions.
        let n_pack = (total_patches + 7) / 8;
        let svf_buffers_reusable = buffers.svf_data_buffer.is_some()
            && buffers.svf_has_veg == has_veg
            && buffers.svf_num_arrays == num_arrays
            && buffers.shadow_u8_n_pack == n_pack;

        if svf_buffers_reusable {
            // Buffers exist with matching config — just clear data and bitpack buffers
            let svf_data_buf = buffers.svf_data_buffer.as_ref().unwrap();
            let shadow_u8_out = buffers.shadow_u8_output_buffer.as_ref().unwrap();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SVF Reinit Encoder"),
                });
            encoder.clear_buffer(svf_data_buf, 0, None);
            encoder.clear_buffer(shadow_u8_out, 0, None);
            self.queue.submit(Some(encoder.finish()));

            eprintln!(
                "[GPU] SVF accumulation reused: {}x{} grid, {} SVF arrays",
                rows, cols, num_arrays,
            );

            return Ok(());
        }

        // Create SVF-specific buffers (first call or config changed)
        let make_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };

        let svf_params_buffer = make_buffer(
            "SVF Accum Params",
            std::mem::size_of::<SvfAccumParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let svf_data_buffer = make_buffer(
            "SVF Data Buffer",
            svf_data_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let svf_result_staging = make_buffer(
            "SVF Result Staging",
            svf_data_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Zero-initialize SVF data buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Init Encoder"),
            });
        encoder.clear_buffer(&svf_data_buffer, 0, None);
        self.queue.submit(Some(encoder.finish()));

        // Create SVF bind group referencing shadow output buffers + SVF buffers
        let svf_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SVF Accumulation Bind Group"),
            layout: &self.svf_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: svf_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: svf_data_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Shadow bitpack buffers (persist across all patch dispatches) ---
        let matrix_bytes = total_pixels * n_pack;
        let matrix_words = (matrix_bytes + 3) / 4; // u32 words
        let num_matrices = if has_veg { 3usize } else { 1usize };
        let packed_output_size = (matrix_words * num_matrices) as u64 * 4; // bytes

        let shadow_u8_params_buffer = make_buffer(
            "Shadow U8 Params",
            std::mem::size_of::<U8PackParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let shadow_u8_output_buffer = make_buffer(
            "Shadow U8 Output",
            packed_output_size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let shadow_u8_staging = make_buffer(
            "Shadow U8 Staging",
            packed_output_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Write static U8 pack params; patch fields are updated per dispatch.
        let u8_params = U8PackParams {
            total_pixels: total_pixels as u32,
            cols: cols as u32,
            rows: rows as u32,
            n_pack: n_pack as u32,
            matrix_words: matrix_words as u32,
            has_veg: if has_veg { 1 } else { 0 },
            patch_byte_idx: 0,
            patch_bit_mask: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.queue.write_buffer(
            &shadow_u8_params_buffer,
            0,
            bytemuck::cast_slice(&[u8_params]),
        );

        // Zero-initialize bitpacked output buffer once before patch loop.
        let mut bitpack_init =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("SVF Bitpack Init Encoder"),
                });
        bitpack_init.clear_buffer(&shadow_u8_output_buffer, 0, None);
        self.queue.submit(Some(bitpack_init.finish()));

        let shadow_u8_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow U8 Pack Bind Group"),
            layout: &self.shadow_u8_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_u8_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.veg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.veg_blocks_bldg_shadow_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_u8_output_buffer.as_entire_binding(),
                },
            ],
        });

        buffers.svf_params_buffer = Some(svf_params_buffer);
        buffers.svf_data_buffer = Some(svf_data_buffer);
        buffers.svf_result_staging = Some(svf_result_staging);
        buffers.svf_bind_group = Some(svf_bind_group);
        buffers.svf_has_veg = has_veg;
        buffers.svf_num_arrays = num_arrays;
        buffers.shadow_u8_params_buffer = Some(shadow_u8_params_buffer);
        buffers.shadow_u8_output_buffer = Some(shadow_u8_output_buffer);
        buffers.shadow_u8_staging = Some(shadow_u8_staging);
        buffers.shadow_u8_bind_group = Some(shadow_u8_bind_group);
        buffers.shadow_u8_packed_size = packed_output_size;
        buffers.shadow_u8_n_pack = n_pack;
        buffers.shadow_u8_matrix_bytes = matrix_bytes;
        buffers.shadow_u8_matrix_words = matrix_words;
        buffers.shadow_u8_num_matrices = num_matrices;

        Ok(())
    }

    /// Per-patch: dispatch shadow + SVF accumulate + bitpack update on GPU (non-blocking).
    ///
    /// Shadow matrices and SVF accumulators stay on GPU for the full patch loop.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_shadow_and_accumulate_svf(
        &self,
        patch_idx: usize,
        azimuth_deg: f32,
        altitude_deg: f32,
        scale: f32,
        max_local_dsm_ht: f32,
        min_sun_elev_deg: f32,
        max_shadow_distance_m: f32,
        weight_iso: f32,
        weight_n: f32,
        weight_e: f32,
        weight_s: f32,
        weight_w: f32,
    ) -> Result<wgpu::SubmissionIndex, String> {
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache empty — call init_svf_accumulation first".to_string())?;

        let svf_params_buf = buffers
            .svf_params_buffer
            .as_ref()
            .ok_or_else(|| "SVF not initialized".to_string())?;
        let u8_params_buf = buffers
            .shadow_u8_params_buffer
            .as_ref()
            .ok_or_else(|| "Shadow U8 params missing".to_string())?;
        let has_veg = buffers.svf_has_veg;
        let rows = buffers.rows;
        let cols = buffers.cols;
        let total_pixels = rows * cols;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;

        // Write shadow params (only thing that changes per patch)
        let azimuth_rad = azimuth_deg.to_radians();
        let altitude_rad = altitude_deg.to_radians();
        let min_sun_elev_rad = min_sun_elev_deg.to_radians();
        let height_reach_m = max_local_dsm_ht / min_sun_elev_rad.tan();
        let max_reach_m = if max_shadow_distance_m > 0.0 { height_reach_m.min(max_shadow_distance_m) } else { height_reach_m };
        let max_index = (max_reach_m / scale).ceil();

        let shadow_params = ShadowParams {
            rows: rows as u32,
            cols: cols as u32,
            azimuth_rad,
            altitude_rad,
            sin_azimuth: azimuth_rad.sin(),
            cos_azimuth: azimuth_rad.cos(),
            tan_azimuth: azimuth_rad.tan(),
            // Guard: f32 tan(~90°) can return a large negative number when
            // the f32 representation of π/2 slightly exceeds the true value.
            // For above-horizon patches, tan must be non-negative; use abs()
            // so the large negative becomes large positive → dz exceeds
            // max_local_dsm_ht immediately → no shadow (physically correct).
            //
            // See shadowing.rs scale note: solweig convention is
            // `scale = pixel_size_m`, so the vertical-drop factor multiplies.
            tan_altitude_by_scale: altitude_rad.tan().abs() * scale,
            scale,
            max_index,
            max_local_dsm_ht,
            has_veg: if has_veg { 1 } else { 0 },
            has_walls: 0,
            _padding: 0,
        };

        self.queue.write_buffer(
            &buffers.params_buffer,
            0,
            bytemuck::cast_slice(&[shadow_params]),
        );

        // Write SVF accumulation params
        let svf_params = SvfAccumParams {
            total_pixels: total_pixels as u32,
            cols: cols as u32,
            rows: rows as u32,
            weight_iso,
            weight_n,
            weight_e,
            weight_s,
            weight_w,
            has_veg: if has_veg { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };

        self.queue
            .write_buffer(svf_params_buf, 0, bytemuck::cast_slice(&[svf_params]));

        let u8_params = U8PackParams {
            total_pixels: total_pixels as u32,
            cols: cols as u32,
            rows: rows as u32,
            n_pack: buffers.shadow_u8_n_pack as u32,
            matrix_words: buffers.shadow_u8_matrix_words as u32,
            has_veg: if has_veg { 1 } else { 0 },
            patch_byte_idx: (patch_idx >> 3) as u32,
            patch_bit_mask: (1u32 << (patch_idx & 7)),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        self.queue
            .write_buffer(u8_params_buf, 0, bytemuck::cast_slice(&[u8_params]));

        // Build command encoder with 3 passes + staging copy
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Shadow+Accum+U8 Encoder"),
            });

        // Reset propagated height buffers (GPU→GPU copy from static inputs)
        encoder.copy_buffer_to_buffer(
            &buffers.dsm_buffer,
            0,
            &buffers.propagated_bldg_height_buffer,
            0,
            buffer_size,
        );
        if has_veg {
            encoder.copy_buffer_to_buffer(
                &buffers.veg_canopy_buffer,
                0,
                &buffers.propagated_veg_height_buffer,
                0,
                buffer_size,
            );
        }

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let (num_workgroups_x, num_workgroups_y) = self.checked_workgroups_2d(
            rows,
            cols,
            workgroup_size_x,
            workgroup_size_y,
            "svf shadow propagation update",
        )?;

        // Pass 1: Shadow propagation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Propagation (SVF)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Pass 2: SVF accumulation (reads shadow outputs, accumulates into svf_data)
        {
            let svf_bg = buffers
                .svf_bind_group
                .as_ref()
                .ok_or("SVF bind group missing")?;
            let svf_workgroup_x = 16u32;
            let svf_workgroup_y = 16u32;
            let (svf_workgroups_x, svf_workgroups_y) = self.checked_workgroups_2d(
                rows,
                cols,
                svf_workgroup_x,
                svf_workgroup_y,
                "svf accumulation",
            )?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SVF Accumulation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.svf_pipeline);
            pass.set_bind_group(0, svf_bg, &[]);
            pass.dispatch_workgroups(svf_workgroups_x, svf_workgroups_y, 1);
        }

        // Pass 3: Update bitpacked shadow matrices for this patch.
        {
            let u8_bg = buffers
                .shadow_u8_bind_group
                .as_ref()
                .ok_or("Shadow U8 bind group missing")?;
            let u8_workgroup_x = 16u32;
            let u8_workgroup_y = 16u32;
            let (u8_workgroups_x, u8_workgroups_y) = self.checked_workgroups_2d(
                rows,
                cols,
                u8_workgroup_x,
                u8_workgroup_y,
                "svf shadow bitpack update",
            )?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Shadow Bitpack Update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shadow_u8_pipeline);
            pass.set_bind_group(0, u8_bg, &[]);
            pass.dispatch_workgroups(u8_workgroups_x, u8_workgroups_y, 1);
        }

        // Submit — no per-patch synchronization; read back once after all patches.
        let submission_index = self.queue.submit(Some(encoder.finish()));
        Ok(submission_index)
    }

    /// Wait for a specific submitted GPU workload to complete.
    pub fn wait_for_submission(
        &self,
        submission_index: wgpu::SubmissionIndex,
    ) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map(|_| ())
            .map_err(|e| format!("GPU poll failed while waiting for SVF dispatch: {:?}", e))
    }

    /// After all patches: read back GPU-built bitpacked shadow matrices.
    pub fn read_svf_bitpacked_shadows(&self) -> Result<SvfBitpackedShadowResult, String> {
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache empty".to_string())?;

        let output_buf = buffers
            .shadow_u8_output_buffer
            .as_ref()
            .ok_or_else(|| "Shadow U8 output buffer missing".to_string())?;
        let staging = buffers
            .shadow_u8_staging
            .as_ref()
            .ok_or_else(|| "Shadow U8 staging not initialized".to_string())?;

        let packed_size = buffers.shadow_u8_packed_size;
        let matrix_bytes = buffers.shadow_u8_matrix_bytes;
        let n_pack = buffers.shadow_u8_n_pack;
        let rows = buffers.rows;
        let cols = buffers.cols;
        let has_veg = buffers.shadow_u8_num_matrices > 1;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Read Bitpacked Shadows Encoder"),
            });
        encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, packed_size);
        let submission_index = self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..packed_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| format!("Poll failed while reading bitpacked shadows: {:?}", e))?;
        receiver
            .recv()
            .map_err(|e| format!("Failed waiting for bitpacked shadow mapping: {}", e))?
            .map_err(|e| format!("Failed to map bitpacked shadow staging: {:?}", e))?;

        let _unmap_guard = MappedBufferGuard::new(staging);
        let data = slice.get_mapped_range();
        let all_bytes: &[u8] = bytemuck::cast_slice(&data);
        let expected = if has_veg {
            matrix_bytes * 3
        } else {
            matrix_bytes
        };
        if all_bytes.len() < expected {
            return Err(format!(
                "Bitpacked shadow buffer too small: got {} bytes, need at least {}",
                all_bytes.len(),
                expected
            ));
        }

        let bldg =
            Array3::from_shape_vec((rows, cols, n_pack), all_bytes[0..matrix_bytes].to_vec())
                .map_err(|e| format!("Failed to reshape bldg shadow matrix: {}", e))?;

        let (veg, vb) = if has_veg {
            let veg_start = matrix_bytes;
            let vb_start = matrix_bytes * 2;
            let veg_arr = Array3::from_shape_vec(
                (rows, cols, n_pack),
                all_bytes[veg_start..vb_start].to_vec(),
            )
            .map_err(|e| format!("Failed to reshape veg shadow matrix: {}", e))?;
            let vb_arr = Array3::from_shape_vec(
                (rows, cols, n_pack),
                all_bytes[vb_start..vb_start + matrix_bytes].to_vec(),
            )
            .map_err(|e| format!("Failed to reshape vb shadow matrix: {}", e))?;
            (veg_arr, vb_arr)
        } else {
            let shape = (rows, cols, n_pack);
            (Array3::<u8>::zeros(shape), Array3::<u8>::zeros(shape))
        };

        Ok(SvfBitpackedShadowResult {
            bldg_sh_matrix: bldg,
            veg_sh_matrix: veg,
            veg_blocks_bldg_sh_matrix: vb,
        })
    }

    /// After all patches: read back accumulated SVF values from GPU.
    pub fn read_svf_results(&self) -> Result<SvfAccumResult, String> {
        let mut cache_guard = self
            .cached
            .lock()
            .map_err(|e| format!("Failed to lock buffer cache: {}", e))?;

        let buffers = cache_guard
            .as_mut()
            .ok_or_else(|| "Buffer cache empty".to_string())?;

        let svf_data_buf = buffers
            .svf_data_buffer
            .as_ref()
            .ok_or_else(|| "SVF not initialized".to_string())?;
        let svf_staging = buffers
            .svf_result_staging
            .as_ref()
            .ok_or_else(|| "SVF staging not initialized".to_string())?;

        let rows = buffers.rows;
        let cols = buffers.cols;
        let total_pixels = rows * cols;
        let has_veg = buffers.svf_has_veg;
        let num_arrays = buffers.svf_num_arrays;
        let buffer_size = (total_pixels * std::mem::size_of::<f32>()) as u64;
        let svf_data_size = buffer_size * num_arrays as u64;

        // Copy svf_data to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SVF Read Results Encoder"),
            });
        encoder.copy_buffer_to_buffer(svf_data_buf, 0, svf_staging, 0, svf_data_size);
        let submission_index = self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = svf_staging.slice(..svf_data_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            })
            .map_err(|e| format!("GPU poll failed while reading SVF results: {:?}", e))?;
        receiver
            .recv()
            .map_err(|e| format!("Failed waiting for SVF result mapping: {}", e))?
            .map_err(|e| format!("Failed to map SVF staging: {:?}", e))?;

        let _unmap_guard = MappedBufferGuard::new(svf_staging);
        let data = slice.get_mapped_range();
        let all: &[f32] = bytemuck::cast_slice(&data);
        let n = total_pixels;

        let extract = |offset: usize, label: &str| -> Result<Array2<f32>, String> {
            Array2::from_shape_vec((rows, cols), all[offset..offset + n].to_vec())
                .map_err(|e| format!("Failed to reshape {} array: {}", label, e))
        };

        let svf = extract(0, "svf")?;
        let svf_n = extract(n, "svf_n")?;
        let svf_e = extract(2 * n, "svf_e")?;
        let svf_s = extract(3 * n, "svf_s")?;
        let svf_w = extract(4 * n, "svf_w")?;

        let (svf_veg, svf_veg_n, svf_veg_e, svf_veg_s, svf_veg_w) = if has_veg {
            (
                Some(extract(5 * n, "svf_veg")?),
                Some(extract(6 * n, "svf_veg_n")?),
                Some(extract(7 * n, "svf_veg_e")?),
                Some(extract(8 * n, "svf_veg_s")?),
                Some(extract(9 * n, "svf_veg_w")?),
            )
        } else {
            (None, None, None, None, None)
        };

        let (svf_aveg, svf_aveg_n, svf_aveg_e, svf_aveg_s, svf_aveg_w) = if has_veg {
            (
                Some(extract(10 * n, "svf_aveg")?),
                Some(extract(11 * n, "svf_aveg_n")?),
                Some(extract(12 * n, "svf_aveg_e")?),
                Some(extract(13 * n, "svf_aveg_s")?),
                Some(extract(14 * n, "svf_aveg_w")?),
            )
        } else {
            (None, None, None, None, None)
        };

        Ok(SvfAccumResult {
            svf,
            svf_n,
            svf_e,
            svf_s,
            svf_w,
            svf_veg,
            svf_veg_n,
            svf_veg_e,
            svf_veg_s,
            svf_veg_w,
            svf_aveg,
            svf_aveg_n,
            svf_aveg_e,
            svf_aveg_s,
            svf_aveg_w,
        })
    }

    #[inline]
    fn write_2d_f32(queue: &wgpu::Queue, buffer: &wgpu::Buffer, arr: &ArrayView2<f32>) {
        if let Some(slice) = arr.as_slice() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(slice));
        } else {
            let packed: Vec<f32> = arr.iter().copied().collect();
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&packed));
        }
    }
}

/// Synchronous wrapper that blocks on async GPU initialization
pub fn create_shadow_gpu_context() -> Result<ShadowGpuContext, String> {
    pollster::block_on(ShadowGpuContext::new())
}
