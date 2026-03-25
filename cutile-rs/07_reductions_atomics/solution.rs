pub fn block_sum(x: &[f32]) -> f32 {
    let mut acc = 0.0;
    for &v in x {
        acc += v;
    }
    acc
}

pub fn block_max(x: &[f32]) -> f32 {
    let mut best = x[0];
    for &v in x.iter().skip(1) {
        if v > best {
            best = v;
        }
    }
    best
}

pub fn histogram_atomic_add(values: &[usize], num_bins: usize) -> Vec<i32> {
    let mut hist = vec![0; num_bins];
    for &b in values {
        hist[b] += 1;
    }
    hist
}
