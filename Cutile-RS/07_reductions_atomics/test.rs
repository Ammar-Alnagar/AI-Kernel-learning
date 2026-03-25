#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_reductions() {
    let sum = if USE_SOLUTION { solution::block_sum(&[1.0, 2.5, -4.0, 8.0]) } else { kernel::block_sum(&[1.0, 2.5, -4.0, 8.0]) };
    assert!((sum - 7.5).abs() < 1e-6);

    let mx = if USE_SOLUTION { solution::block_max(&[1.0, 2.5, -4.0, 8.0]) } else { kernel::block_max(&[1.0, 2.5, -4.0, 8.0]) };
    assert!((mx - 8.0).abs() < 1e-6);

    let h = if USE_SOLUTION { solution::histogram_atomic_add(&[0, 1, 1, 3, 0, 2, 3, 3], 4) } else { kernel::histogram_atomic_add(&[0, 1, 1, 3, 0, 2, 3, 3], 4) };
    assert_eq!(h, vec![2, 2, 1, 3]);
}
