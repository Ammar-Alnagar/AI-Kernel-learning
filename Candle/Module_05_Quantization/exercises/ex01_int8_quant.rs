use candle_core::{Device, Result, Tensor};

fn fake_int8_per_channel(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let scale = (w.abs()?.max_keepdim(1)? / 127.0)?;
    let q = w.broadcast_div(&scale)?.round()?.clamp(-127f64, 127f64)?;
    Ok((q, scale))
}

fn dequant(q: &Tensor, scale: &Tensor) -> Result<Tensor> {
    q.broadcast_mul(scale)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 0.5f32, (512, 512), &dev)?;
    let (q, s) = fake_int8_per_channel(&w)?;
    let w_hat = dequant(&q, &s)?;
    let mse = (&w - &w_hat)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    println!("mse={mse:.6}");
    Ok(())
}
