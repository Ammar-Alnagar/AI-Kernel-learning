#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

#[test]
fn test_persistent() {
    let ids = if USE_SOLUTION { solution::persistent_tile_ids(2, 3, 10) } else { kernel::persistent_tile_ids(2, 3, 10) };
    assert_eq!(ids, vec![2, 5, 8]);

    let x: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let y: Vec<f32> = vec![1.0; 20];
    let out0 = if USE_SOLUTION { solution::persistent_vector_add(&x, &y, 4, 0, 2) } else { kernel::persistent_vector_add(&x, &y, 4, 0, 2) };
    let out1 = if USE_SOLUTION { solution::persistent_vector_add(&x, &y, 4, 1, 2) } else { kernel::persistent_vector_add(&x, &y, 4, 1, 2) };
    let out: Vec<f32> = out0.iter().zip(out1.iter()).map(|(a, b)| a + b).collect();
    let expected: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
    assert_eq!(out, expected);
}
