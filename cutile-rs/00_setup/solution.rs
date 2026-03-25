pub fn ceil_div(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}

pub fn vector_add_block(x: &[f32], y: &[f32], out: &mut [f32], block_id: usize, block_size: usize) {
    for lane in 0..block_size {
        let idx = block_id * block_size + lane;
        if idx < x.len() {
            out[idx] = x[idx] + y[idx];
        }
    }
}

pub fn vector_add_launch(x: &[f32], y: &[f32], block_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let num_blocks = ceil_div(x.len(), block_size);
    for block_id in 0..num_blocks {
        vector_add_block(x, y, &mut out, block_id, block_size);
    }
    out
}
