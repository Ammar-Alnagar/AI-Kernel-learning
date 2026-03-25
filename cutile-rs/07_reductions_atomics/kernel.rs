pub fn block_sum(x: &[f32]) -> f32 {
    let mut acc = 0.0;
    for i in 0..x.len() {
        // FILL IN
        todo!("fill block_sum")
    }
    acc
}

pub fn block_max(x: &[f32]) -> f32 {
    let mut best = x[0];
    for i in 1..x.len() {
        // FILL IN
        todo!("fill block_max")
    }
    best
}

pub fn histogram_atomic_add(values: &[usize], num_bins: usize) -> Vec<i32> {
    let mut hist = vec![0; num_bins];
    for &b in values {
        // FILL IN
        todo!("fill histogram_atomic_add")
    }
    hist
}
