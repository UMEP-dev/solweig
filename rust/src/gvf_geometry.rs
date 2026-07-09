//! GVF geometry caching — precompute building ray-trace once per DSM.
//!
//! `f` (building occlusion) is binary (0/1) and monotonically descending.
//! We represent it as a blocking distance (u16) per pixel per azimuth.
//! All purely-geometric accumulators are precomputed and cached.

use ndarray::{s, Array1, Array2, ArrayView2, Zip};
use rayon::prelude::*;

const PI: f32 = std::f32::consts::PI;

/// Per-azimuth precomputed geometry.
pub(crate) struct AzimuthGeometry {
    /// Azimuth angle in degrees for this geometry snapshot.
    pub azimuth_deg: f32,
    /// Step at which each pixel gets blocked (f→0). `second` if never blocked.
    pub blocking_distance: Array2<u16>,
    /// (dx, dy) shift offsets per step.
    pub shifts: Vec<(isize, isize)>,
    /// Wall-facing mask for this azimuth direction.
    pub facesh: Array2<f32>,
    /// Accumulated albedo (no shadow) through occlusion — snapshot at `first` threshold.
    pub albnosh_accum_first: Array2<f32>,
    /// Accumulated albedo (no shadow) through occlusion — full range.
    pub albnosh_accum: Array2<f32>,
    /// Wall albedo (no shadow) weighted by geometric wall visibility — snapshot at `first`.
    pub wallnosh_accum_first: Array2<f32>,
    /// Wall albedo (no shadow) weighted by geometric wall visibility — full range.
    pub wallnosh_accum: Array2<f32>,
    /// Whether any wall is geometrically visible within `first` height.
    pub wall_influence_first: Array2<f32>,
    /// Whether any wall is geometrically visible within full height.
    pub wall_influence: Array2<f32>,
}

/// Full GVF geometry cache for all 18 azimuths.
pub(crate) struct GvfGeometryCache {
    pub azimuths: Vec<AzimuthGeometry>,
    pub first: f32,
    pub second: f32,
    /// Cached gvfalbnosh outputs (purely geometric): center, E, S, W, N.
    pub cached_albnosh: Array2<f32>,
    pub cached_albnosh_e: Array2<f32>,
    pub cached_albnosh_s: Array2<f32>,
    pub cached_albnosh_w: Array2<f32>,
    pub cached_albnosh_n: Array2<f32>,
}

/// Compute (dx, dy) shift for a given azimuth and step index.
fn compute_shift(azimuth_rad: f32, index: f32) -> (isize, isize) {
    let pibyfour = PI / 4.;
    let threetimespibyfour = 3. * pibyfour;
    let fivetimespibyfour = 5. * pibyfour;
    let seventimespibyfour = 7. * pibyfour;
    let sinazimuth = azimuth_rad.sin();
    let cosazimuth = azimuth_rad.cos();
    let tanazimuth = azimuth_rad.tan();
    let signsinazimuth = sinazimuth.signum();
    let signcosazimuth = cosazimuth.signum();

    let (dx, dy) = if (pibyfour..threetimespibyfour).contains(&azimuth_rad)
        || (fivetimespibyfour..seventimespibyfour).contains(&azimuth_rad)
    {
        (
            -1. * signcosazimuth * (index / tanazimuth).abs().round(),
            signsinazimuth * index,
        )
    } else {
        (
            -1. * signcosazimuth * index,
            signsinazimuth * (index * tanazimuth).abs().round(),
        )
    };

    (dx as isize, dy as isize)
}

/// Compute slice bounds for a shift (dx, dy) on a grid of size (sizex, sizey).
/// Returns (x_c_slice, x_p_slice) as ((xc1,xc2,yc1,yc2), (xp1,xp2,yp1,yp2)).
fn compute_slices(
    dx: isize,
    dy: isize,
    sizex: usize,
    sizey: usize,
) -> ((isize, isize, isize, isize), (isize, isize, isize, isize)) {
    let absdx = dx.abs();
    let absdy = dy.abs();

    let xc1 = (dx + absdx) / 2;
    let xc2 = sizex as isize + (dx - absdx) / 2;
    let yc1 = (dy + absdy) / 2;
    let yc2 = sizey as isize + (dy - absdy) / 2;

    let xp1 = -(dx - absdx) / 2;
    let xp2 = sizex as isize - (dx + absdx) / 2;
    let yp1 = -(dy - absdy) / 2;
    let yp2 = sizey as isize - (dy + absdy) / 2;

    ((xc1, xc2, yc1, yc2), (xp1, xp2, yp1, yp2))
}

/// Compute facesh mask for a given azimuth vs wall aspects.
fn compute_facesh(
    azimuth_rad: f32,
    wall_aspect: ArrayView2<f32>,
    wall_ht: ArrayView2<f32>,
) -> Array2<f32> {
    let azilow = azimuth_rad - PI / 2.;
    let azihigh = azimuth_rad + PI / 2.;
    let wallbol = wall_ht.mapv(|x| if x > 0. { 1. } else { 0. });

    if azilow >= 0. && azihigh < 2. * PI {
        let mut facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect < azilow || aspect >= azihigh {
                1.
            } else {
                0.
            }
        });
        facesh = facesh - &wallbol + 1.;
        facesh
    } else if azilow < 0. && azihigh <= 2. * PI {
        let azilow_adj = azilow + 2. * PI;
        let mut facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect > azilow_adj || aspect <= azihigh {
                -1.
            } else {
                0.
            }
        });
        facesh.mapv_inplace(|x| x + 1.);
        facesh
    } else {
        let azihigh_adj = azihigh - 2. * PI;
        let mut facesh = Zip::from(wall_aspect).map_collect(|&aspect| {
            if aspect > azilow || aspect <= azihigh_adj {
                -1.
            } else {
                0.
            }
        });
        facesh.mapv_inplace(|x| x + 1.);
        facesh
    }
}

/// Precompute geometry for a single azimuth direction.
fn precompute_azimuth_geometry(
    azimuth_deg: f32,
    buildings: ArrayView2<f32>,
    wall_aspect: ArrayView2<f32>,
    wall_ht: ArrayView2<f32>,
    alb_grid: ArrayView2<f32>,
    wall_albedo: f32,
    first: f32,
    second: f32,
) -> AzimuthGeometry {
    let (sizex, sizey) = (buildings.nrows(), buildings.ncols());
    let azimuth_rad = azimuth_deg * (PI / 180.);

    // Precompute shifts
    let num_steps = second as usize;
    let mut shifts = Vec::with_capacity(num_steps);
    for n in 0..num_steps {
        shifts.push(compute_shift(azimuth_rad, n as f32));
    }

    // Ray-trace: compute blocking distances and geometric accumulators
    let mut f = buildings.to_owned();
    let mut blocking_distance = Array2::<u16>::from_elem((sizex, sizey), second as u16);
    let mut tempbu = Array2::<f32>::zeros((sizex, sizey));
    let mut tempalbnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut tempbubwall = Array2::<f32>::zeros((sizex, sizey));

    let mut weightsumalbnosh = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwallnosh = Array2::<f32>::zeros((sizex, sizey));

    let mut weightsumalbnosh_first = Array2::<f32>::zeros((sizex, sizey));
    let mut weightsumalbwallnosh_first = Array2::<f32>::zeros((sizex, sizey));

    for (n, &(dx, dy)) in shifts.iter().enumerate() {
        let ((xc1, xc2, yc1, yc2), (xp1, xp2, yp1, yp2)) = compute_slices(dx, dy, sizex, sizey);
        let x_c_slice = s![xc1..xc2, yc1..yc2];
        let x_p_slice = s![xp1..xp2, yp1..yp2];

        // Shift buildings and update occlusion
        tempbu
            .slice_mut(x_p_slice)
            .assign(&buildings.slice(x_c_slice));
        Zip::from(f.view_mut())
            .and(tempbu.view())
            .for_each(|f_val, &tb| {
                *f_val = f_val.min(tb);
            });

        // Record blocking distance: first step where f drops to 0
        Zip::from(&mut blocking_distance)
            .and(&f)
            .for_each(|bd, &fv| {
                // Only update if not already blocked (bd still at initial value or higher)
                if fv == 0. && *bd > n as u16 {
                    *bd = n as u16;
                }
            });

        // Accumulate albedo (no shadow) weighted by f
        tempalbnosh
            .slice_mut(x_p_slice)
            .assign(&alb_grid.slice(x_c_slice));
        Zip::from(&mut weightsumalbnosh)
            .and(&tempalbnosh)
            .and(&f)
            .for_each(|w, &a, &fv| *w += a * fv);

        // Wall tracking: tempbubwall = "have we seen any wall?" latch
        Zip::from(&mut tempbubwall).and(&f).for_each(|bubw, &fv| {
            let bwall = 1. - fv;
            *bubw = if *bubw + bwall > 0. { 1. } else { 0. };
        });
        weightsumalbwallnosh.zip_mut_with(&tempbubwall, |w, &b| *w += b * wall_albedo);

        // Snapshot at first-height threshold
        if (n + 1) as f32 <= first {
            weightsumalbnosh_first.assign(&weightsumalbnosh);
            weightsumalbwallnosh_first.assign(&weightsumalbwallnosh);
        }
    }

    // Wall influence masks
    let wall_influence_first = weightsumalbwallnosh_first.mapv(|x| (x > 0.) as i32 as f32);
    let wall_influence = weightsumalbwallnosh.mapv(|x| (x > 0.) as i32 as f32);

    // Facesh mask
    let facesh = compute_facesh(azimuth_rad, wall_aspect, wall_ht);

    AzimuthGeometry {
        azimuth_deg,
        blocking_distance,
        shifts,
        facesh,
        albnosh_accum_first: weightsumalbnosh_first,
        albnosh_accum: weightsumalbnosh,
        wallnosh_accum_first: weightsumalbwallnosh_first,
        wallnosh_accum: weightsumalbwallnosh,
        wall_influence_first,
        wall_influence,
    }
}

/// Precompute GVF geometry cache for all 18 azimuths.
///
/// This runs the building ray-trace once and caches the results.
/// Subsequent timesteps skip the geometry and only compute thermal quantities.
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_gvf_geometry_cpu(
    buildings: ArrayView2<f32>,
    wall_aspect: ArrayView2<f32>,
    wall_ht: ArrayView2<f32>,
    alb_grid: ArrayView2<f32>,
    pixel_scale: f32,
    first_ht: f32,
    second_ht: f32,
    wall_albedo: f32,
) -> GvfGeometryCache {
    let (rows, cols) = (buildings.nrows(), buildings.ncols());
    // Metres → pixels: pixel_scale is metres per pixel (see sun_on_surface).
    // Clamped to the grid extent so out-of-range shifts cannot panic the
    // slice arithmetic on coarse rasters.
    let first = (first_ht / pixel_scale).round().max(1.);
    let second = (second_ht / pixel_scale)
        .round()
        .min(rows.max(cols) as f32);

    let azimuth_a: Array1<f32> = Array1::range(5.0, 359.0, 20.0);
    let num_azimuths = azimuth_a.len() as f32;
    let num_azimuths_half = num_azimuths / 2.0;

    // Precompute per-azimuth geometry in parallel
    let az_geoms: Vec<AzimuthGeometry> = azimuth_a
        .iter()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&&az| {
            precompute_azimuth_geometry(
                az,
                buildings,
                wall_aspect,
                wall_ht,
                alb_grid,
                wall_albedo,
                first,
                second,
            )
        })
        .collect();

    // Accumulate cached gvfalbnosh outputs (5 directions) from per-azimuth results
    let buildings_inv = buildings.mapv(|x| 1. - x);
    let scale_all = 1.0 / num_azimuths;
    let scale_half = 1.0 / num_azimuths_half;

    let mut albnosh_center = Array2::<f32>::zeros((rows, cols));
    let mut albnosh_e = Array2::<f32>::zeros((rows, cols));
    let mut albnosh_s = Array2::<f32>::zeros((rows, cols));
    let mut albnosh_w = Array2::<f32>::zeros((rows, cols));
    let mut albnosh_n = Array2::<f32>::zeros((rows, cols));

    for (i, geom) in az_geoms.iter().enumerate() {
        let azimuth = azimuth_a[i];

        // Per-azimuth gvfalbnosh (matches sun_on_surface post-loop logic)
        let gvfalbnosh1 = (&geom.wallnosh_accum_first + &geom.albnosh_accum_first) / (first + 1.)
            * &geom.wall_influence_first
            + &geom.albnosh_accum_first / first * geom.wall_influence_first.mapv(|x| 1. - x);
        let gvfalbnosh2 = (&geom.wallnosh_accum + &geom.albnosh_accum) / second
            * &geom.wall_influence
            + &geom.albnosh_accum / second * geom.wall_influence.mapv(|x| 1. - x);

        let gvfalbnosh_az = (&gvfalbnosh1 * 0.5 + &gvfalbnosh2 * 0.4) / 0.9 * &buildings
            + &alb_grid * &buildings_inv;

        albnosh_center += &gvfalbnosh_az;

        if (0.0..180.0).contains(&azimuth) {
            albnosh_e += &gvfalbnosh_az;
        }
        if (90.0..270.0).contains(&azimuth) {
            albnosh_s += &gvfalbnosh_az;
        }
        if (180.0..360.0).contains(&azimuth) {
            albnosh_w += &gvfalbnosh_az;
        }
        if !(90.0..270.0).contains(&azimuth) {
            albnosh_n += &gvfalbnosh_az;
        }
    }

    // Scale by number of azimuths
    albnosh_center.mapv_inplace(|v| v * scale_all);
    albnosh_e.mapv_inplace(|v| v * scale_half);
    albnosh_s.mapv_inplace(|v| v * scale_half);
    albnosh_w.mapv_inplace(|v| v * scale_half);
    albnosh_n.mapv_inplace(|v| v * scale_half);

    GvfGeometryCache {
        azimuths: az_geoms,
        first,
        second,
        cached_albnosh: albnosh_center,
        cached_albnosh_e: albnosh_e,
        cached_albnosh_s: albnosh_s,
        cached_albnosh_w: albnosh_w,
        cached_albnosh_n: albnosh_n,
    }
}

/// GPU-accelerated variant of [`precompute_gvf_geometry_cpu`].
///
/// Runs the per-pixel/per-azimuth building march on the GPU (one thread per
/// output pixel, all azimuths looped internally) and assembles a
/// `GvfGeometryCache` identical in layout to the CPU path. `first`/`second`
/// are derived with the exact same expression as the CPU path so both agree
/// bit-for-bit. The per-azimuth albedo accumulator fields
/// (`albnosh_accum*`, `wallnosh_accum*`, `wall_influence*`) are write-only
/// after the in-shader reduction and are left empty here.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_gvf_geometry_gpu(
    ctx: &crate::gpu::GvfPrecomputeGpuContext,
    buildings: ArrayView2<f32>,
    wall_aspect: ArrayView2<f32>,
    wall_ht: ArrayView2<f32>,
    alb_grid: ArrayView2<f32>,
    pixel_scale: f32,
    first_ht: f32,
    second_ht: f32,
    wall_albedo: f32,
) -> Result<GvfGeometryCache, String> {
    let (rows, cols) = (buildings.nrows(), buildings.ncols());
    let first = (first_ht / pixel_scale).round().max(1.);
    let second = (second_ht / pixel_scale)
        .round()
        .min(rows.max(cols) as f32);

    let azimuth_a: Array1<f32> = Array1::range(5.0, 359.0, 20.0);
    let num_azimuths = azimuth_a.len();
    let num_steps = second as usize;
    if num_steps == 0 {
        return Err("GVF precompute: second resolves to zero steps".to_string());
    }

    // Precompute per-azimuth shifts on the CPU (identical to the CPU path);
    // the shader consumes them, and they are also stored in the cache for the
    // CPU runtime fallback.
    let shifts: Vec<Vec<(isize, isize)>> = azimuth_a
        .iter()
        .map(|&az| {
            let az_rad = az * (PI / 180.);
            (0..num_steps)
                .map(|n| compute_shift(az_rad, n as f32))
                .collect()
        })
        .collect();

    let azimuths_deg: Vec<f32> = azimuth_a.to_vec();

    let raw = ctx.precompute(
        buildings,
        wall_aspect,
        wall_ht,
        alb_grid,
        &azimuths_deg,
        &shifts,
        num_steps,
        first,
        second,
        wall_albedo,
    )?;

    let rc = rows * cols;
    if raw.albnosh.len() != 5 * rc
        || raw.blocking_distance.len() != num_azimuths * rc
        || raw.facesh.len() != num_azimuths * rc
    {
        return Err(format!(
            "GVF precompute GPU output size mismatch: albnosh={}, bd={}, facesh={} (expected {}/{}/{})",
            raw.albnosh.len(),
            raw.blocking_distance.len(),
            raw.facesh.len(),
            5 * rc,
            num_azimuths * rc,
            num_azimuths * rc,
        ));
    }

    let chan = |i: usize| -> Result<Array2<f32>, String> {
        Array2::from_shape_vec((rows, cols), raw.albnosh[i * rc..(i + 1) * rc].to_vec())
            .map_err(|e| format!("albnosh channel {}: {}", i, e))
    };
    let cached_albnosh = chan(0)?;
    let cached_albnosh_e = chan(1)?;
    let cached_albnosh_s = chan(2)?;
    let cached_albnosh_w = chan(3)?;
    let cached_albnosh_n = chan(4)?;

    let empty = || Array2::<f32>::zeros((0, 0));
    let mut az_geoms = Vec::with_capacity(num_azimuths);
    for (i, az_shifts) in shifts.into_iter().enumerate() {
        let offset = i * rc;
        let bd_u16: Vec<u16> = raw.blocking_distance[offset..offset + rc]
            .iter()
            .map(|&v| v as u16)
            .collect();
        let blocking_distance = Array2::from_shape_vec((rows, cols), bd_u16)
            .map_err(|e| format!("blocking_distance az {}: {}", i, e))?;
        let facesh =
            Array2::from_shape_vec((rows, cols), raw.facesh[offset..offset + rc].to_vec())
                .map_err(|e| format!("facesh az {}: {}", i, e))?;

        az_geoms.push(AzimuthGeometry {
            azimuth_deg: azimuth_a[i],
            blocking_distance,
            shifts: az_shifts,
            facesh,
            albnosh_accum_first: empty(),
            albnosh_accum: empty(),
            wallnosh_accum_first: empty(),
            wallnosh_accum: empty(),
            wall_influence_first: empty(),
            wall_influence: empty(),
        });
    }

    Ok(GvfGeometryCache {
        azimuths: az_geoms,
        first,
        second,
        cached_albnosh,
        cached_albnosh_e,
        cached_albnosh_s,
        cached_albnosh_w,
        cached_albnosh_n,
    })
}
