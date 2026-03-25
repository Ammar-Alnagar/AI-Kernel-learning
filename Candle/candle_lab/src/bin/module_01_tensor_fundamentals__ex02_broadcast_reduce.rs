use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;

    let x = Tensor::randn(0f32, 1f32, (3, 4), &dev)?;
    let bias = Tensor::zeros((4,), DType::F32, &dev)?;

    let y = x.broadcast_add(&bias)?;
    let mean = y.mean_keepdim(1)?;

    let ex2 = y.sqr()?.mean_keepdim(1)?;
    let var = (&ex2 - mean.sqr()?)?;

    println!("y shape: {:?}", y.shape());
    println!("mean shape: {:?}", mean.shape());
    println!("var shape: {:?}", var.shape());
    Ok(())
}
