pub fn program_range_1d(pid: usize, block_size: usize) -> (usize, usize) {
    // FILL IN
    todo!("fill program_range_1d")
}

pub fn lane_offsets(pid: usize, block_size: usize) -> Vec<usize> {
    // FILL IN
    todo!("fill lane_offsets")
}

pub fn copy_kernel_block(src: &[f32], dst: &mut [f32], pid: usize, block_size: usize) {
    let offs = lane_offsets(pid, block_size);
    for idx in offs {
        // FILL IN
        todo!("fill copy_kernel_block")
    }
}

pub fn copy_launch(src: &[f32], block_size: usize) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    // FILL IN
    todo!("fill copy_launch")
}
