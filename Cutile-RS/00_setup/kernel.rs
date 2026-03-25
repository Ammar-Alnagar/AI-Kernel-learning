pub fn ceil_div(n: usize, d: usize) -> usize {
    // FILL IN
    todo!("fill ceil_div")
}

pub fn vector_add_block(x: &[f32], y: &[f32], out: &mut [f32], block_id: usize, block_size: usize) {
    for lane in 0..block_size {
        let idx = block_id * block_size + lane;
        // FILL IN
        todo!("fill vector_add_block")
    }
}

pub fn vector_add_launch(x: &[f32], y: &[f32], block_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let num_blocks = ceil_div(x.len(), block_size);
    for block_id in 0..num_blocks {
        // FILL IN
        todo!("fill vector_add_launch")
    }
    out
}
