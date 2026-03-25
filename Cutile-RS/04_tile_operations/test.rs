#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_tile_ops() {
    let y = if USE_SOLUTION { solution::fused_affine(&[1.0, 2.0, 3.0], 2.0, -1.0) } else { kernel::fused_affine(&[1.0, 2.0, 3.0], 2.0, -1.0) };
    assert_eq!(y, vec![1.0, 3.0, 5.0]);

    let mat = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
    let out = if USE_SOLUTION { solution::row_bias_add(&mat, &[0.5, -1.0]) } else { kernel::row_bias_add(&mat, &[0.5, -1.0]) };
    assert_eq!(out, vec![vec![0.5, 0.0], vec![2.5, 2.0]]);

    let tr = if USE_SOLUTION { solution::transpose_2d(&vec![vec![1, 2, 3], vec![4, 5, 6]]) } else { kernel::transpose_2d(&vec![vec![1, 2, 3], vec![4, 5, 6]]) };
    assert_eq!(tr, vec![vec![1, 4], vec![2, 5], vec![3, 6]]);
}
