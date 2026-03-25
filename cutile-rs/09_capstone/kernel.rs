pub fn attention_scores(q: &[Vec<f32>], k: &[Vec<f32>], scale: f32) -> Vec<Vec<f32>> {
    // FILL IN
    todo!("fill attention_scores")
}

pub fn causal_mask_inplace(scores: &mut [Vec<f32>], mask_value: f32) {
    for i in 0..scores.len() {
        for j in 0..scores.len() {
            // FILL IN
            todo!("fill causal_mask_inplace")
        }
    }
}

pub fn row_softmax(x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut y = vec![vec![0.0; x[0].len()]; x.len()];
    for i in 0..x.len() {
        // FILL IN
        todo!("fill row_softmax")
    }
    y
}

pub fn fused_attention(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32, causal: bool) -> Vec<Vec<f32>> {
    let mut scores = attention_scores(q, k, scale);
    if causal {
        causal_mask_inplace(&mut scores, -1e9);
    }
    let probs = row_softmax(&scores);
    // FILL IN
    todo!("fill fused_attention")
}
