use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn bench(dev: &Device, m: usize, n: usize, k: usize, dt: DType) -> Result<f64> {
    let a = Tensor::randn(0f32, 1f32, (m, k), dev)?.to_dtype(dt)?;
    let b = Tensor::randn(0f32, 1f32, (k, n), dev)?.to_dtype(dt)?;
    for _ in 0..10 {
        let _ = a.matmul(&b)?;
    }
    let iters = 50;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = a.matmul(&b)?;
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    Ok(flops / (ms * 1e-3) / 1e12)
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    for dt in [DType::F32, DType::F16, DType::BF16] {
        let tflops = bench(&dev, 4096, 4096, 4096, dt)?;
        println!("{dt:?} => {tflops:.2} TFLOPS");
    }
    Ok(())
}
