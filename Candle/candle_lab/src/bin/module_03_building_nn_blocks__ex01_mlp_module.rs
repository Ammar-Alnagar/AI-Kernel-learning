use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

struct Mlp {
    l1: Linear,
    l2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_dim: usize, hidden: usize, out_dim: usize) -> Result<Self> {
        let l1 = candle_nn::linear(in_dim, hidden, vb.pp("l1"))?;
        let l2 = candle_nn::linear(hidden, out_dim, vb.pp("l2"))?;
        Ok(Self { l1, l2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.l2.forward(&self.l1.forward(x)?.relu()?)
    }
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let mlp = Mlp::new(vb, 16, 32, 4)?;
    let x = Tensor::randn(0f32, 1f32, (2, 16), &dev)?;
    let y = mlp.forward(&x)?;
    println!("shape={:?}", y.shape());
    Ok(())
}
