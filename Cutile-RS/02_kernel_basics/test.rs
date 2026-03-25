#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_kernel_basics() {
    let range = if USE_SOLUTION { solution::program_range_1d(3, 8) } else { kernel::program_range_1d(3, 8) };
    assert_eq!(range, (24, 32));

    let offs = if USE_SOLUTION { solution::lane_offsets(2, 4) } else { kernel::lane_offsets(2, 4) };
    assert_eq!(offs, vec![8, 9, 10, 11]);

    let src: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let mut dst = vec![0.0; 10];
    if USE_SOLUTION { solution::copy_kernel_block(&src, &mut dst, 2, 4) } else { kernel::copy_kernel_block(&src, &mut dst, 2, 4) }
    assert_eq!(&dst[8..10], &src[8..10]);

    let out = if USE_SOLUTION { solution::copy_launch(&src, 4) } else { kernel::copy_launch(&src, 4) };
    assert_eq!(out, src);
}
