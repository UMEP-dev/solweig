// GVF cached thermal accumulation — GPU compute shader.
//
// Replaces the sun_on_surface_cached() × 18 azimuths loop from
// gvf_calc_with_cache() (gvf.rs). Each thread handles one pixel and
// iterates over all azimuths, marching along precomputed ray shifts
// using cached blocking_distance to skip the building ray-trace.
//
// Inputs (per timestep):
//   lup         — differential upwelling longwave (excess above ambient)
//   albshadow   — albedo × shadow
//   sunwall_mask — sun-on-wall indicator (>0 → sun hits wall)
//
// Inputs (cached, upload once per DSM):
//   blocking_distance — step at which each pixel gets occluded [18 × R × C]
//   facesh            — wall-facing mask per azimuth            [18 × R × C]
//   shifts            — (dx, dy) offsets per azimuth × step     [18 × max_steps]
//
// Outputs (10 channels, raw accumulated — Rust adds scaling + baselines):
//   [0] lup_total   [1] alb_total
//   [2] lup_e       [3] alb_e
//   [4] lup_s       [5] alb_s
//   [6] lup_w       [7] alb_w
//   [8] lup_n       [9] alb_n

// ── Uniform parameters ──────────────────────────────────────────────────

struct Params {
    rows:          u32,
    cols:          u32,
    num_azimuths:  u32,
    max_steps:     u32,
    first:         f32,
    second:        f32,
    lwall:         f32,
    wall_albedo:   f32,
};

struct AzimuthMeta {
    dir_mask:      u32,    // bit0=E, bit1=S, bit2=W, bit3=N
    shift_offset:  u32,    // offset into shifts[] for this azimuth
    _pad0:         u32,
    _pad1:         u32,
};

// ── Bind group 0: params + azimuth info + shifts ────────────────────────

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> azimuth_info: array<AzimuthMeta>;
@group(0) @binding(2) var<storage, read> shifts: array<vec2<i32>>;

// ── Bind group 1: cached geometry (upload once per DSM) ─────────────────

@group(1) @binding(0) var<storage, read> blocking_distance: array<u32>;
@group(1) @binding(1) var<storage, read> facesh: array<f32>;

// ── Bind group 2: per-timestep inputs + outputs ─────────────────────────

@group(2) @binding(0) var<storage, read> lup: array<f32>;
@group(2) @binding(1) var<storage, read> albshadow: array<f32>;
@group(2) @binding(2) var<storage, read> sunwall_mask: array<f32>;
@group(2) @binding(3) var<storage, read_write> outputs: array<f32>;

// ── Helpers ─────────────────────────────────────────────────────────────

fn pixel_idx(row: u32, col: u32) -> u32 {
    return row * params.cols + col;
}

fn geom_idx(az: u32, row: u32, col: u32) -> u32 {
    return az * params.rows * params.cols + row * params.cols + col;
}

fn out_idx(channel: u32, row: u32, col: u32) -> u32 {
    return channel * params.rows * params.cols + row * params.cols + col;
}

// ── Main compute kernel ─────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let rows_i = i32(params.rows);
    let cols_i = i32(params.cols);
    let row_i = i32(row);
    let col_i = i32(col);

    // Accumulate across all azimuths
    var total_lup: f32 = 0.0;
    var total_alb: f32 = 0.0;
    var dir_lup_e: f32 = 0.0;
    var dir_alb_e: f32 = 0.0;
    var dir_lup_s: f32 = 0.0;
    var dir_alb_s: f32 = 0.0;
    var dir_lup_w: f32 = 0.0;
    var dir_alb_w: f32 = 0.0;
    var dir_lup_n: f32 = 0.0;
    var dir_alb_n: f32 = 0.0;

    for (var az = 0u; az < params.num_azimuths; az++) {
        let az_meta = azimuth_info[az];
        let bd = blocking_distance[geom_idx(az, row, col)];
        let fsh = facesh[geom_idx(az, row, col)];

        // Per-azimuth accumulators (matches sun_on_surface_cached)
        var tempbub: f32 = 0.0;
        var cur_lup: f32 = 0.0;
        var cur_alb: f32 = 0.0;
        var cur_wallsun: f32 = 0.0;
        var wsLupsh: f32 = 0.0;
        var wsalbsh: f32 = 0.0;
        var wsLwall: f32 = 0.0;
        var wsalbwall: f32 = 0.0;
        var wswall: f32 = 0.0;

        // First-height snapshots
        var wsLupsh_1: f32 = 0.0;
        var wsalbsh_1: f32 = 0.0;
        var wsLwall_1: f32 = 0.0;
        var wsalbwall_1: f32 = 0.0;
        var wswall_1: f32 = 0.0;

        for (var n = 0u; n < params.max_steps; n++) {
            let shift = shifts[az_meta.shift_offset + n];
            let src_row = row_i + shift.x;
            let src_col = col_i + shift.y;

            // Refresh the shifted values while the source pixel is in bounds.
            // Out of bounds they keep the last value (EDGE CLAMP, do NOT skip):
            // sunonsurface_2018a never clears its shift buffers, and
            // gvf_precompute.wgsl builds blocking_distance the same way.
            if (src_row >= 0 && src_row < rows_i &&
                src_col >= 0 && src_col < cols_i) {
                let src_idx = pixel_idx(u32(src_row), u32(src_col));
                cur_lup = lup[src_idx];
                cur_alb = albshadow[src_idx];
                cur_wallsun = sunwall_mask[src_idx];
            }

            if (n < bd) {
                wsLupsh += cur_lup;
                wsalbsh += cur_alb;
                // Wall-sun latch
                if ((tempbub + cur_wallsun) > 0.0) {
                    tempbub = 1.0;
                }
            }

            // Wall accumulators update every step using the current latch state
            wsLwall += tempbub * params.lwall;
            wsalbwall += tempbub * params.wall_albedo;
            wswall += tempbub;

            // Snapshot at first-height threshold
            if (f32(n + 1u) <= params.first) {
                wswall_1 = wswall;
                wsLwall_1 = wsLwall;
                wsLupsh_1 = wsLupsh;
                wsalbwall_1 = wsalbwall;
                wsalbsh_1 = wsalbsh;
            }
        }

        // ── Post-loop: two-layer blending (matches sun_on_surface_cached) ──

        let wsi_1 = select(0.0, 1.0, wswall_1 > 0.0);
        let wsi_2 = select(0.0, 1.0, wswall > 0.0);

        // Keep correction (uses facesh from geometry cache)
        let keep_raw = select(0.0, 1.0, wswall == params.second) - fsh;
        let keep = select(keep_raw, 0.0, keep_raw == -1.0);

        // ── gvfLup ──

        let gvfLup1 = ((wsLwall_1 + wsLupsh_1) / (params.first + 1.0)) * wsi_1
                    + (wsLupsh_1 / params.first) * (1.0 - wsi_1);

        var Lwall_adj = wsLwall;
        if (keep == 1.0) {
            Lwall_adj = 0.0;
        }
        let gvfLup2 = ((Lwall_adj + wsLupsh) / (params.second + 1.0)) * wsi_2
                    + (wsLupsh / params.second) * (1.0 - wsi_2);

        let gvfLup = (gvfLup1 * 0.5 + gvfLup2 * 0.4) / 0.9;

        // ── gvfalb ──

        let gvfalb1 = ((wsalbwall_1 + wsalbsh_1) / (params.first + 1.0)) * wsi_1
                    + (wsalbsh_1 / params.first) * (1.0 - wsi_1);

        var albwall_adj = wsalbwall;
        if (keep == 1.0) {
            albwall_adj = 0.0;
        }
        let gvfalb2 = ((albwall_adj + wsalbsh) / (params.second + 1.0)) * wsi_2
                    + (wsalbsh / params.second) * (1.0 - wsi_2);

        let gvfalb = (gvfalb1 * 0.5 + gvfalb2 * 0.4) / 0.9;

        // ── Accumulate into total and directional bins ──

        total_lup += gvfLup;
        total_alb += gvfalb;

        let mask = az_meta.dir_mask;
        if ((mask & 1u) != 0u) { dir_lup_e += gvfLup; dir_alb_e += gvfalb; }
        if ((mask & 2u) != 0u) { dir_lup_s += gvfLup; dir_alb_s += gvfalb; }
        if ((mask & 4u) != 0u) { dir_lup_w += gvfLup; dir_alb_w += gvfalb; }
        if ((mask & 8u) != 0u) { dir_lup_n += gvfLup; dir_alb_n += gvfalb; }
    }

    // Write raw accumulated values (Rust applies scaling + baselines)
    outputs[out_idx(0u, row, col)] = total_lup;
    outputs[out_idx(1u, row, col)] = total_alb;
    outputs[out_idx(2u, row, col)] = dir_lup_e;
    outputs[out_idx(3u, row, col)] = dir_alb_e;
    outputs[out_idx(4u, row, col)] = dir_lup_s;
    outputs[out_idx(5u, row, col)] = dir_alb_s;
    outputs[out_idx(6u, row, col)] = dir_lup_w;
    outputs[out_idx(7u, row, col)] = dir_alb_w;
    outputs[out_idx(8u, row, col)] = dir_lup_n;
    outputs[out_idx(9u, row, col)] = dir_alb_n;
}
