#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_capstone_shapes_and_softmax() {
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![vec![2.0, 3.0], vec![5.0, 7.0]];
    let scale = 1.0;

    let mut scores = if USE_SOLUTION { solution::attention_scores(&q, &k, scale) } else { kernel::attention_scores(&q, &k, scale) };
    assert_eq!(scores.len(), 2);
    if USE_SOLUTION { solution::causal_mask_inplace(&mut scores, -1e9) } else { kernel::causal_mask_inplace(&mut scores, -1e9) }
    assert!(scores[0][1] < -1e8);

    let probs = if USE_SOLUTION { solution::row_softmax(&scores) } else { kernel::row_softmax(&scores) };
    for row in probs {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    let out = if USE_SOLUTION { solution::fused_attention(&q, &k, &v, scale, true) } else { kernel::fused_attention(&q, &k, &v, scale, true) };
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].len(), 2);
}
