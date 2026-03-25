pub fn fused_affine(x: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        y[i] = alpha * x[i] + beta;
    }
    y
}

pub fn row_bias_add(x: &[Vec<f32>], bias: &[f32]) -> Vec<Vec<f32>> {
    let rows = x.len();
    let cols = x[0].len();
    let mut y = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            y[r][c] = x[r][c] + bias[c];
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
            y[c][r] = x[r][c];
        }
    }
    y
}
