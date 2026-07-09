// GVF geometry precompute — GPU compute shader.
//
// Ports `precompute_gvf_geometry` (gvf_geometry.rs) to the GPU. One thread
// per output pixel loops all azimuths (5..359 step 20) and, inside each,
// marches the precomputed ray shifts. Per (pixel, azimuth) it reproduces the
// CPU building ray-trace to emit:
//   - blocking_distance (u32) : first step the ray hits a non-building
//   - facesh (f32)            : wall-facing mask
//   - albedo-no-shadow accumulators, blended and reduced in-shader into the
//     5 cached_albnosh_* channels (center, E, S, W, N).
//
// CRITICAL parity note: the CPU folds whole-grid temp buffers that retain the
// LAST in-bounds sample once a ray leaves the raster (edge clamp). This shader
// reproduces that by keeping `tb`/`ta` at their last in-bounds value instead of
// skipping out-of-bounds steps.
//
// Outputs (Rust reads them back and assembles GvfGeometryCache):
//   albnosh_out : [5 x R x C] f32  (already scaled by 1/naz or 1/(naz/2))
//   bd_out      : [naz x R x C] u32
//   facesh_out  : [naz x R x C] f32

const PI: f32 = 3.14159265358979;

// ── Uniform parameters ──────────────────────────────────────────────────

struct Params {
    rows:         u32,
    cols:         u32,
    num_azimuths: u32,
    max_steps:    u32,
    first:        f32,
    second:       f32,
    wall_albedo:  f32,
    _pad0:        u32,
};

struct AzimuthMeta {
    dir_mask:     u32,    // bit0=E, bit1=S, bit2=W, bit3=N
    shift_offset: u32,    // offset into shifts[] for this azimuth
    azimuth_rad:  f32,    // az_deg * PI/180 (computed CPU-side, bit-identical)
    _pad1:        u32,
};

// ── Bind group 0: params + azimuth info + shifts ────────────────────────

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> azimuth_info: array<AzimuthMeta>;
@group(0) @binding(2) var<storage, read> shifts: array<vec2<i32>>;

// ── Bind group 1: static per-DSM inputs ─────────────────────────────────

@group(1) @binding(0) var<storage, read> buildings: array<f32>;
@group(1) @binding(1) var<storage, read> alb_grid: array<f32>;
@group(1) @binding(2) var<storage, read> wall_aspect: array<f32>;
@group(1) @binding(3) var<storage, read> wall_ht: array<f32>;

// ── Bind group 2: outputs ───────────────────────────────────────────────

@group(2) @binding(0) var<storage, read_write> albnosh_out: array<f32>;
@group(2) @binding(1) var<storage, read_write> bd_out: array<u32>;
@group(2) @binding(2) var<storage, read_write> facesh_out: array<f32>;

// ── Helpers ─────────────────────────────────────────────────────────────

fn pixel_idx(row: u32, col: u32) -> u32 {
    return row * params.cols + col;
}

fn geom_idx(az: u32, row: u32, col: u32) -> u32 {
    return az * params.rows * params.cols + row * params.cols + col;
}

fn facesh_value(az_rad: f32, aspect: f32, wh: f32) -> f32 {
    let half_pi = PI * 0.5;
    let two_pi = PI * 2.0;
    let azilow = az_rad - half_pi;
    let azihigh = az_rad + half_pi;
    let wallbol = select(0.0, 1.0, wh > 0.0);

    if (azilow >= 0.0 && azihigh < two_pi) {
        let base = select(0.0, 1.0, aspect < azilow || aspect >= azihigh);
        return base - wallbol + 1.0;
    } else if (azilow < 0.0 && azihigh <= two_pi) {
        let lo = azilow + two_pi;
        let base = select(0.0, -1.0, aspect > lo || aspect <= azihigh);
        return base + 1.0;
    } else {
        let hi = azihigh - two_pi;
        let base = select(0.0, -1.0, aspect > azilow || aspect <= hi);
        return base + 1.0;
    }
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

    let here = pixel_idx(row, col);
    let b_here = buildings[here];
    let binv = 1.0 - b_here;
    let alb_here = alb_grid[here];
    let aspect_here = wall_aspect[here];
    let wh_here = wall_ht[here];

    let first = params.first;
    let second = params.second;
    let wall_albedo = params.wall_albedo;

    // Cross-azimuth reduction accumulators (5 directional channels).
    var acc_center: f32 = 0.0;
    var acc_e: f32 = 0.0;
    var acc_s: f32 = 0.0;
    var acc_w: f32 = 0.0;
    var acc_n: f32 = 0.0;

    for (var az = 0u; az < params.num_azimuths; az++) {
        let az_meta = azimuth_info[az];

        // ── March along this azimuth's ray ──
        var f: f32 = b_here;               // building occlusion (monotone descend)
        var bd: u32 = u32(second);         // blocking distance, default "never"
        var tb: f32 = 0.0;                 // stale building sample (edge clamp)
        var ta: f32 = 0.0;                 // stale albedo sample (edge clamp)
        var tempbubwall: f32 = 0.0;        // wall-seen latch

        var wsalbnosh: f32 = 0.0;
        var wsalbwall: f32 = 0.0;
        var wsalbnosh_first: f32 = 0.0;
        var wsalbwall_first: f32 = 0.0;

        for (var n = 0u; n < params.max_steps; n++) {
            let shift = shifts[az_meta.shift_offset + n];
            let src_row = row_i + shift.x;
            let src_col = col_i + shift.y;

            if (src_row >= 0 && src_row < rows_i &&
                src_col >= 0 && src_col < cols_i) {
                let sidx = pixel_idx(u32(src_row), u32(src_col));
                tb = buildings[sidx];      // update stale buffers
                ta = alb_grid[sidx];
            }
            // else: keep stale tb, ta (EDGE CLAMP — do NOT skip)

            f = min(f, tb);
            if (f == 0.0 && bd > n) {
                bd = n;
            }
            wsalbnosh += ta * f;

            let bwall = 1.0 - f;
            if ((tempbubwall + bwall) > 0.0) {
                tempbubwall = 1.0;
            } else {
                tempbubwall = 0.0;
            }
            wsalbwall += tempbubwall * wall_albedo;

            if (f32(n + 1u) <= first) {
                wsalbnosh_first = wsalbnosh;
                wsalbwall_first = wsalbwall;
            }
        }

        // ── Per-azimuth geometry outputs ──
        bd_out[geom_idx(az, row, col)] = bd;
        facesh_out[geom_idx(az, row, col)] =
            facesh_value(az_meta.azimuth_rad, aspect_here, wh_here);

        // ── Blend (matches precompute_gvf_geometry reduction) ──
        let wi_first = select(0.0, 1.0, wsalbwall_first > 0.0);
        let wi = select(0.0, 1.0, wsalbwall > 0.0);

        let g1 = (wsalbwall_first + wsalbnosh_first) / (first + 1.0) * wi_first
               + (wsalbnosh_first / first) * (1.0 - wi_first);
        let g2 = (wsalbwall + wsalbnosh) / second * wi
               + (wsalbnosh / second) * (1.0 - wi);

        let gaz = (g1 * 0.5 + g2 * 0.4) / 0.9 * b_here + alb_here * binv;

        acc_center += gaz;
        let mask = az_meta.dir_mask;
        if ((mask & 1u) != 0u) { acc_e += gaz; }
        if ((mask & 2u) != 0u) { acc_s += gaz; }
        if ((mask & 4u) != 0u) { acc_w += gaz; }
        if ((mask & 8u) != 0u) { acc_n += gaz; }
    }

    // ── Scale and write reduced cached_albnosh channels ──
    let naz = f32(params.num_azimuths);
    let scale_all = 1.0 / naz;
    let scale_half = 1.0 / (naz / 2.0);
    let rc = params.rows * params.cols;

    albnosh_out[0u * rc + here] = acc_center * scale_all;
    albnosh_out[1u * rc + here] = acc_e * scale_half;
    albnosh_out[2u * rc + here] = acc_s * scale_half;
    albnosh_out[3u * rc + here] = acc_w * scale_half;
    albnosh_out[4u * rc + here] = acc_n * scale_half;
}
