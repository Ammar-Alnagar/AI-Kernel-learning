#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_matmul() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let c = if USE_SOLUTION { solution::matmul_naive(&a, &b) } else { kernel::matmul_naive(&a, &b) };
    assert_eq!(c, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);

    let mut acc = vec![vec![0.0; 2]; 2];
    if USE_SOLUTION { solution::mma_tile_update(&mut acc, &a, &b) } else { kernel::mma_tile_update(&mut acc, &a, &b) }
    assert_eq!(acc, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
}
