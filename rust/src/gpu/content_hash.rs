//! Full-content hash of a static GPU input raster.
//!
//! The shadow GPU context skips re-uploading inputs that have not changed
//! since the last dispatch. "Not changed" has to mean the values, not the
//! address: a new surface's arrays routinely land on the address a freed
//! surface used. This hash reads every element. It runs on every timestep, so
//! it is built for throughput: parallel chunks, four independent mixing
//! chains per chunk.
//!
//! The hash is not cryptographic. It only has to make two different rasters
//! collide with negligible probability. A contiguous view and a strided view
//! of the same values may hash differently, which costs one extra upload and
//! nothing else.

use ndarray::ArrayView2;
use rayon::prelude::*;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Bytes hashed per rayon task.
const CHUNK_BYTES: usize = 1 << 18;
/// Independent mixing chains per task.
const LANES: usize = 4;

#[inline]
fn mix(hash: u64, word: u64) -> u64 {
    let x = (hash ^ word).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^ (x >> 32)
}

#[inline]
fn word_at(bytes: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

fn hash_chunk(bytes: &[u8]) -> u64 {
    let mut lanes = [SEED, SEED ^ 0x1111, SEED ^ 0x2222, SEED ^ 0x3333];
    let mut blocks = bytes.chunks_exact(8 * LANES);
    for block in &mut blocks {
        for (lane, hash) in lanes.iter_mut().enumerate() {
            *hash = mix(*hash, word_at(block, lane * 8));
        }
    }
    let mut hash = lanes.iter().fold(SEED, |acc, &lane| mix(acc, lane));
    let mut words = blocks.remainder().chunks_exact(8);
    for word in &mut words {
        hash = mix(hash, word_at(word, 0));
    }
    for &byte in words.remainder() {
        hash = mix(hash, byte as u64);
    }
    hash
}

/// Order-sensitive hash of a byte slice, computed in parallel chunks.
fn hash_bytes(bytes: &[u8]) -> u64 {
    let parts: Vec<u64> = bytes.par_chunks(CHUNK_BYTES).map(hash_chunk).collect();
    parts.into_iter().fold(mix(SEED, bytes.len() as u64), mix)
}

pub(crate) fn hash_2d_f32(arr: &ArrayView2<f32>) -> u64 {
    match arr.as_slice() {
        Some(slice) => hash_bytes(bytemuck::cast_slice(slice)),
        None => arr
            .iter()
            .fold(mix(SEED, arr.len() as u64), |hash, v| mix(hash, v.to_bits() as u64)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn one_changed_element_changes_the_hash() {
        let mut a = Array2::<f32>::zeros((300, 400));
        let before = hash_2d_f32(&a.view());
        a[[211, 137]] = 12.0;
        assert_ne!(before, hash_2d_f32(&a.view()));
    }

    #[test]
    fn equal_content_at_different_addresses_hashes_equal() {
        let a = Array2::<f32>::from_shape_fn((120, 90), |(r, c)| (r * 7 + c) as f32);
        let b = a.clone();
        assert_eq!(hash_2d_f32(&a.view()), hash_2d_f32(&b.view()));
    }

    #[test]
    fn swapped_elements_change_the_hash() {
        let mut a = Array2::<f32>::from_shape_fn((50, 50), |(r, c)| (r * 50 + c) as f32);
        let before = hash_2d_f32(&a.view());
        a.swap([3, 4], [40, 41]);
        assert_ne!(before, hash_2d_f32(&a.view()));
    }
}
