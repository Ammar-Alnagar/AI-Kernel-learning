#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_load_store() {
    let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let out = if USE_SOLUTION { solution::masked_pad_load(&x, 8, 8) } else { kernel::masked_pad_load(&x, 8, 8) };
    assert_eq!(out, vec![8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut dst = vec![0.0; 6];
    if USE_SOLUTION { solution::masked_store(&mut dst, 4, &[1.0, 2.0, 3.0, 4.0]) } else { kernel::masked_store(&mut dst, 4, &[1.0, 2.0, 3.0, 4.0]) }
    assert_eq!(dst, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);

    let gathered = if USE_SOLUTION { solution::gather_1d(&[10.0, 20.0, 30.0, 40.0, 50.0], &[4, 0, 3, 1]) } else { kernel::gather_1d(&[10.0, 20.0, 30.0, 40.0, 50.0], &[4, 0, 3, 1]) };
    assert_eq!(gathered, vec![50.0, 10.0, 40.0, 20.0]);

    let mut dst2 = vec![0.0; 4];
    if USE_SOLUTION { solution::scatter_add_1d(&mut dst2, &[0, 1, 1, 3], &[2.0, 3.0, 4.0, 5.0]) } else { kernel::scatter_add_1d(&mut dst2, &[0, 1, 1, 3], &[2.0, 3.0, 4.0, 5.0]) }
    assert_eq!(dst2, vec![2.0, 7.0, 0.0, 5.0]);
}
