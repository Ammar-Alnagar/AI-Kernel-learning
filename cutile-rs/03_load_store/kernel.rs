pub fn masked_pad_load(x: &[f32], block_start: usize, block_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; block_size];
    for lane in 0..block_size {
        let idx = block_start + lane;
        // FILL IN
        todo!("fill masked_pad_load")
    }
    out
}

pub fn masked_store(dst: &mut [f32], block_start: usize, values: &[f32]) {
    for lane in 0..values.len() {
        let idx = block_start + lane;
        // FILL IN
        todo!("fill masked_store")
    }
}

pub fn gather_1d(src: &[f32], indices: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0; indices.len()];
    for i in 0..indices.len() {
        // FILL IN
        todo!("fill gather_1d")
    }
    out
}

pub fn scatter_add_1d(dst: &mut [f32], indices: &[usize], values: &[f32]) {
    for i in 0..indices.len() {
        // FILL IN
        todo!("fill scatter_add_1d")
    }
}
