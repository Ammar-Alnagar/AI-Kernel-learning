pub fn attention_scores(q: &[Vec<f32>], k: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
    let s = q.len();
    let d = q[0].len();
    let mut out = vec![vec![0.0; s]; s];
    for i in 0..s {
        for j in 0..s {
            let mut acc = 0.0;
            for kk in 0..d {
                acc += q[i][kk] * k[j][kk];
            }
            out[i][j] = scale * acc;
        }
    }
    out
}

pub fn causal_mask_inplace(scores: &mut [Vec<f32>], mask_value: f32) {
    for i in 0..scores.len() {
        for j in 0..scores.len() {
            if j > i {
                scores[i][j] = mask_value;
            }
        }
    }
}

pub fn row_softmax(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut y = vec![vec![0.0; x[0].len()]; x.len()];
    for i in 0..x.len() {
        let mut maxv = f32::NEG_INFINITY;
        for &v in &x[i] {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0;
        for j in 0..x[i].len() {
            let e = (x[i][j] - maxv).exp();
            y[i][j] = e;
            sum += e;
        }
        for j in 0..x[i].len() {
            y[i][j] /= sum;
        }
    }
    y
}

pub fn fused_attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32, causal: bool) -> Vec<Vec<f32>> {
    let mut scores = attention_scores(q, k, scale);
    if causal {
        causal_mask_inplace(&mut scores, -1e9);
    }
    let probs = row_softmax(&scores);
    matmul(&probs, v)
}

fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let m = a.len();
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[i][kk] * b[kk][j];
            }
            c[i][j] = acc;
        }
    }
    c
}
