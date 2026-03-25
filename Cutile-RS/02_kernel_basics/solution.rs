pub fn program_range_1d(pid: usize, block_size: usize) -> (usize, usize) {
    let start = pid * block_size;
    (start, start + block_size)
}

pub fn lane_offsets(pid: usize, block_size: usize) -> Vec<usize> {
    let (start, _) = program_range_1d(pid, block_size);
    (0..block_size).map(|i| start + i).collect()
}

pub fn copy_kernel_block(src: &[f32], dst: &mut [f32], pid: usize, block_size: usize) {
    let offs = lane_offsets(pid, block_size);
    for idx in offs {
        if idx < src.len() {
            dst[idx] = src[idx];
        }
    }
}

pub fn copy_launch(src: &[f32], block_size: usize) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    let num_programs = (src.len() + block_size - 1) / block_size;
    for pid in 0..num_programs {
        copy_kernel_block(src, &mut dst, pid, block_size);
    }
    dst
}
