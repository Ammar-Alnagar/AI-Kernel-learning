use candle_core::{Device, Result, Tensor};
use candle_nn::ops;

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dims3()?.2 as f64;
    let logits = (q.matmul(&k.transpose(1, 2)?)? / d.sqrt())?;
    let p = ops::softmax(&logits, 2)?;
    p.matmul(v)
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
