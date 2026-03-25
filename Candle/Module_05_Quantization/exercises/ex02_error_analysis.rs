use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 0.5f32, (1024, 1024), &dev)?;
    let mse = w.sqr()?.mean_all()?.to_scalar::<f32>()?;
    println!("baseline second moment={mse:.6}");
    println!("extend this file to compare per-tensor vs per-channel quantization error.");
    Ok(())
}
