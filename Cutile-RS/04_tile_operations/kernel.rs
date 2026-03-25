pub fn fused_affine(x: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        // FILL IN
        todo!("fill fused_affine")
    }
    y
}

pub fn row_bias_add(x: &[Vec<f32>], bias: &[f32]) -> Vec<Vec<f32>> {
    let rows = x.len();
    let cols = x[0].len();
    let mut y = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            // FILL IN
            todo!("fill row_bias_add")
        }
    }
    y
}

pub fn transpose_2d(x: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let rows = x.len();
    let cols = x[0].len();
    let mut y = vec![vec![0; rows]; cols];
    for r in 0..rows {
        for c in 0..cols {
            // FILL IN
            todo!("fill transpose_2d")
        }
    }
    y
}
