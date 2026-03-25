use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

fn bench(dev: &Device, m: usize, n: usize, k: usize, dt: DType) -> Result<f64> {
    let a = Tensor::randn(0f32, 1f32, (m, k), dev)?.to_dtype(dt)?;
    let b = Tensor::randn(0f32, 1f32, (k, n), dev)?.to_dtype(dt)?;

    // TODO: warmup + timed loop
    let ms = _____;

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    Ok(flops / (ms * 1e-3) / 1e12)
}

fn main() -> Result<()> {
    // TODO: create CUDA device
    let dev = _____;

    for dt in [DType::F32, DType::F16, DType::BF16] {
        let tflops = bench(&dev, 4096, 4096, 4096, dt)?;
        println!("{dt:?} => {tflops:.2} TFLOPS");
    }
    Ok(())
}
