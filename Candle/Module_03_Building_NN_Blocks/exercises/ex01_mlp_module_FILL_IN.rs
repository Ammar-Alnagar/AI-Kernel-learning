use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

struct Mlp {
    l1: Linear,
    l2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, in_dim: usize, hidden: usize, out_dim: usize) -> Result<Self> {
        // TODO
        let l1 = _____;
        let l2 = _____;
        Ok(Self { l1, l2 })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO: linear -> relu -> linear
        _____
    }
}

fn main() -> Result<()> {
    let _ = Device::Cpu;
    Ok(())
}
