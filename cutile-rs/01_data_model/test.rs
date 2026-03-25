#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_data_model() {
    let offset = if USE_SOLUTION { solution::offset_2d(3, 5, 16, 1) } else { kernel::offset_2d(3, 5, 16, 1) };
    assert_eq!(offset, 53);

    let mat: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let rows = vec![0, 2, 1, 2];
    let cols = vec![1, 3, 0, 2];
    let out = if USE_SOLUTION {
        solution::gather_2d_strided(&mat, &rows, &cols, 4, 1)
    } else {
        kernel::gather_2d_strided(&mat, &rows, &cols, 4, 1)
    };
    assert_eq!(out, vec![1.0, 11.0, 4.0, 10.0]);

    let casted = if USE_SOLUTION { solution::cast_array_i32(&[1.2, -3.9, 8.1]) } else { kernel::cast_array_i32(&[1.2, -3.9, 8.1]) };
    assert_eq!(casted, vec![1, -3, 8]);

    let p2 = if USE_SOLUTION { solution::is_power_of_two(16) } else { kernel::is_power_of_two(16) };
    let np2 = if USE_SOLUTION { solution::is_power_of_two(18) } else { kernel::is_power_of_two(18) };
    assert!(p2);
    assert!(!np2);
}
