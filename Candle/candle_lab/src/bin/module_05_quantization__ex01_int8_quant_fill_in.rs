use candle_core::{Device, Result, Tensor};

fn fake_int8_per_channel(w: &Tensor) -> Result<(Tensor, Tensor)> {
    // TODO: scale = max(abs(w), dim=1, keepdim=true) / 127
    let scale = _____;

    // TODO: q = round(w / scale).clamp(-127, 127)
    let q = _____;
    Ok((q, scale))
}

fn dequant(q: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // TODO
    _____
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 0.5f32, (512, 512), &dev)?;
    let (q, s) = fake_int8_per_channel(&w)?;
    let w_hat = dequant(&q, &s)?;
    println!("{:?}", w_hat.shape());
    Ok(())
}
