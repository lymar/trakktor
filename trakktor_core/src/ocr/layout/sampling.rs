//! The host half of the deformable sampler, shared by both runtimes.
//!
//! Bilinear sampling at arbitrary positions — `grid_sample` — exists in
//! neither candle nor burn, so both runtimes gather the four neighbours
//! themselves. *Which* four and with what weights is pure index arithmetic, and
//! it is computed here, once, so the two runtimes cannot drift apart on the
//! rounding of a corner.

/// The geometry of one flattened feature level.
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub height: usize,
    pub width: usize,
    /// Where the level starts in the concatenated sequence of positions.
    pub start: usize,
}

impl Level {
    pub fn count(&self) -> usize { self.height * self.width }
}

/// The four neighbours of every sample, and how much each contributes.
///
/// `grid` is `heads × samples` positions in `0..1` of the level's own width and
/// height, heads one after another; `count` is how many positions one head's
/// slice of the value tensor holds, so an index can address the whole
/// heads-flattened tensor at once.
///
/// Both arrays are laid out corner by corner: `indices[c * rows + at]` is the
/// `c`-th neighbour of sample `at`.
///
/// A corner outside the map gets weight zero and an index pointing at its own
/// head's first row — the `zeros` padding mode of `grid_sample`, expressed
/// without a branch in the gather. Replicating the edge instead would let a
/// query looking past the border see the border twice.
pub fn corners(
    grid: &[f32],
    level: &Level,
    count: usize,
    heads: usize,
) -> (Vec<u32>, Vec<f32>) {
    let rows = grid.len() / 2;
    let samples = rows / heads;
    let (height, width) = (level.height as f32, level.width as f32);
    let mut indices = vec![0u32; 4 * rows];
    let mut blend = vec![0f32; 4 * rows];
    for at in 0..rows {
        // The half-pixel shift is `align_corners: false`: column zero of a map
        // `width` wide is centred at `0.5 / width` in normalized coordinates.
        let x = grid[2 * at] * width - 0.5;
        let y = grid[2 * at + 1] * height - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let (fx, fy) = (x - x0, y - y0);
        let head = at / samples;
        for (corner, (dx, dy)) in
            [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
                .iter()
                .enumerate()
        {
            let cx = x0 + dx;
            let cy = y0 + dy;
            let inside = cx >= 0.0 && cy >= 0.0 && cx < width && cy < height;
            let wx = if *dx == 0.0 { 1.0 - fx } else { fx };
            let wy = if *dy == 0.0 { 1.0 - fy } else { fy };
            indices[corner * rows + at] = (head * count) as u32 +
                if inside {
                    (cy as u32) * level.width as u32 + cx as u32
                } else {
                    0
                };
            blend[corner * rows + at] = if inside { wx * wy } else { 0.0 };
        }
    }
    (indices, blend)
}
