use candle_core::{Device, Result, Tensor};
use candle_nn::ops;

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // TODO: logits = q @ k^T / sqrt(d)
    let logits = _____;

    // TODO: causal mask
    let masked = _____;

    // TODO: probs then output
    let p = ops::softmax(&masked, 2)?;
    let out = _____;
    Ok(out)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (2, 16, 64), &dev)?;
    let k = Tensor::randn(0f32, 1f32, (2, 16, 64), &dev)?;
    let v = Tensor::randn(0f32, 1f32, (2, 16, 64), &dev)?;
    let o = attention(&q, &k, &v)?;
    println!("{:?}", o.shape());
    Ok(())
}
