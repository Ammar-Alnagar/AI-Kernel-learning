#[path = "kernel.rs"]
mod kernel;
#[path = "solution.rs"]
mod solution;

const USE_SOLUTION: bool = option_env!("TUTORIAL_USE_SOLUTION").is_some();

fn ceil_div(n: usize, d: usize) -> usize {
    if USE_SOLUTION { solution::ceil_div(n, d) } else { kernel::ceil_div(n, d) }
}

fn vector_add_block(x: &[f32], y: &[f32], out: &mut [f32], block_id: usize, block_size: usize) {
    if USE_SOLUTION {
        solution::vector_add_block(x, y, out, block_id, block_size)
    } else {
        kernel::vector_add_block(x, y, out, block_id, block_size)
    }
}

fn vector_add_launch(x: &[f32], y: &[f32], block_size: usize) -> Vec<f32> {
    if USE_SOLUTION { solution::vector_add_launch(x, y, block_size) } else { kernel::vector_add_launch(x, y, block_size) }
}

#[test]
fn test_setup() {
    assert_eq!(ceil_div(17, 8), 3);
    let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let y: Vec<f32> = (0..10).map(|v| (2 * v) as f32).collect();
    let mut out = vec![0.0; 10];
    vector_add_block(&x, &y, &mut out, 1, 4);
    assert_eq!(&out[4..8], &[12.0, 15.0, 18.0, 21.0]);
    let z = vector_add_launch(&x, &y, 4);
    for i in 0..x.len() {
        assert!((z[i] - (x[i] + y[i])).abs() < 1e-6);
    }
}
