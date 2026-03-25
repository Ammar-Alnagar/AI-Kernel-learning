pub fn matmul_naive(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                // FILL IN
                todo!("fill matmul_naive")
            }
            c[i][j] = acc;
        }
    }
    c
}

pub fn mma_tile_update(acc: &mut [Vec<f32>], a_tile: &[Vec<f32>], b_tile: &[Vec<f32>]) {
    for i in 0..a_tile.len() {
        for j in 0..b_tile[0].len() {
            for kk in 0..a_tile[0].len() {
                // FILL IN
                todo!("fill mma_tile_update")
            }
        }
    }
}
