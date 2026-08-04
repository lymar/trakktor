//! Host-side numeric tables shared by the runtimes.
//!
//! Three computations feed the networks with plain `f32` rows built on the
//! host: the vision tower's learned position grid interpolated onto the
//! input's patch grid, the tower's two-dimensional rotary angles, and the
//! decoder's three-axis rotary angles. Each is pure arithmetic over a handful
//! of kilobytes, and each must come out **identical** whichever runtime runs
//! the tensors — a runtime that rebuilt them its own way could drift by a
//! rounding and turn a token-exact comparison into noise. So they live here
//! once, and the runtimes only lift the results onto their devices.

/// The base of the tower's two-dimensional rotary embedding. Not the decoder's
/// `rope_theta` — the tower has its own, and it is not in the config file.
pub const TOWER_ROPE_THETA: f64 = 10_000.0;

/// The position vectors for a `height × width` patch grid, interpolated from
/// the learned `side × side` grid and laid out one row per patch, row-major.
///
/// The sample points are `linspace(0, side − 1, n)` in each direction, so the
/// corners of the learned grid land exactly on the corners of the target one
/// and everything between is a bilinear blend of four neighbours. Two
/// published implementations disagree here; this is the `transformers` way,
/// the reference every measurement was taken against.
pub fn interpolate_positions(
    positions: &[f32],
    side: usize,
    dim: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let axis = |n: usize| -> Vec<(usize, usize, f32)> {
        (0..n)
            .map(|i| {
                let at = if n > 1 {
                    i as f32 * (side - 1) as f32 / (n - 1) as f32
                } else {
                    0.0
                };
                let low = at as usize;
                let high = (low + 1).min(side - 1);
                (low, high, at - low as f32)
            })
            .collect()
    };
    let rows = axis(height);
    let columns = axis(width);

    let mut out = vec![0f32; height * width * dim];
    for (y, &(top, bottom, fy)) in rows.iter().enumerate() {
        for (x, &(left, right, fx)) in columns.iter().enumerate() {
            let corners = [
                ((top * side + left) * dim, (1.0 - fy) * (1.0 - fx)),
                ((top * side + right) * dim, (1.0 - fy) * fx),
                ((bottom * side + left) * dim, fy * (1.0 - fx)),
                ((bottom * side + right) * dim, fy * fx),
            ];
            let at = (y * width + x) * dim;
            let slot = &mut out[at..at + dim];
            for (base, weight) in corners {
                for (value, source) in
                    slot.iter_mut().zip(&positions[base..base + dim])
                {
                    *value += source * weight;
                }
            }
        }
    }
    out
}

/// The tower's two-dimensional rotary angles for a patch grid, one row per
/// patch, `[patches, head_dim / 2]`: the patch's row angles fill the first
/// quarter of the head dimension and its column angles the second. The
/// rotation pairs channel `i` with channel `i + head_dim / 2`, so each angle
/// is carried once.
pub fn tower_angles(grid: (usize, usize, usize), head_dim: usize) -> Vec<f32> {
    let (_, height, width) = grid;
    let quarter = head_dim / 4;
    let inverse: Vec<f32> = (0..quarter)
        .map(|i| {
            (1.0 / TOWER_ROPE_THETA
                .powf(2.0 * i as f64 / (head_dim / 2) as f64))
                as f32
        })
        .collect();

    let half = head_dim / 2;
    let mut angles = Vec::with_capacity(height * width * half);
    for row in 0..height {
        for column in 0..width {
            angles.extend(inverse.iter().map(|f| row as f32 * f));
            angles.extend(inverse.iter().map(|f| column as f32 * f));
        }
    }
    angles
}

/// The decoder's three-axis rotary angles for `positions`, one row per
/// position, `[seq, head_dim / 2]`: the rotation pairs channel `i` with
/// channel `i + head_dim / 2`, so each angle is carried once. Within a row
/// the channels are split between the three axes by `section`.
pub fn decoder_angles(
    positions: &[[i64; 3]],
    section: &[usize],
    rope_theta: f64,
    head_dim: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let inverse: Vec<f32> = (0..half)
        .map(|i| {
            (1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64)) as f32
        })
        .collect();

    let mut angles = vec![0f32; positions.len() * half];
    for (row, axes) in positions.iter().enumerate() {
        let slot = &mut angles[row * half..(row + 1) * half];
        let mut at = 0;
        for (&size, &position) in section.iter().zip(axes.iter()) {
            for (channel, angle) in slot[at..at + size].iter_mut().enumerate() {
                *angle = position as f32 * inverse[at + channel];
            }
            at += size;
        }
    }
    angles
}
