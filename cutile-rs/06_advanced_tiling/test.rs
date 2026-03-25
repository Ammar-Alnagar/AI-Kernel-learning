#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_advanced_tiling() {
    let v = if USE_SOLUTION { solution::linear_pid_to_2d(7, 3) } else { kernel::linear_pid_to_2d(7, 3) };
    assert_eq!(v, (2, 1));

    let m = if USE_SOLUTION { solution::grouped_pid_to_2d(5, 5, 4, 2) } else { kernel::grouped_pid_to_2d(5, 5, 4, 2) };
    assert_eq!(m, (1, 2));

    let b = if USE_SOLUTION { solution::tile_bounds(2, 1, 4, 8, 10, 19) } else { kernel::tile_bounds(2, 1, 4, 8, 10, 19) };
    assert_eq!(b, ((8, 10), (8, 16)));
}
