use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;

    let x = Tensor::randn(0f32, 1f32, (3, 4), &dev)?;
    let bias = Tensor::zeros((4,), DType::F32, &dev)?;

    // TODO: y = x + bias
    let y = _____;

    // TODO: row mean over dim=1 keepdim=true
    let mean = _____;

    // TODO: variance = E[x^2] - E[x]^2
    let ex2 = _____;
    let var = _____;

    println!("y shape: {:?}", y.shape());
    println!("mean shape: {:?}", mean.shape());
    println!("var shape: {:?}", var.shape());
    Ok(())
}
