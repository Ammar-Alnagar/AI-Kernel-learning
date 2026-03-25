use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{loss, ops, optim::AdamW, Linear, Module, VarBuilder, VarMap};

struct Mlp { l1: Linear, l2: Linear }
impl Mlp {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            l1: candle_nn::linear(2, 32, vb.pp("l1"))?,
            l2: candle_nn::linear(32, 2, vb.pp("l2"))?,
        })
    }
}
impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> { self.l2.forward(&self.l1.forward(x)?.relu()?) }
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (512, 2), &dev)?;
    let y = x.i((.., 0))?.gt(0f32)?.to_dtype(DType::U32)?;

    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let net = Mlp::new(vb)?;
    let mut opt = AdamW::new_lr(vm.all_vars(), 1e-3)?;

    for step in 0..200 {
        let logits = net.forward(&x)?;
        let ce = loss::cross_entropy(&logits, &y)?;
        opt.backward_step(&ce)?;
        if step % 50 == 0 {
            let pred = ops::softmax(&logits, 1)?.argmax(1)?;
            let acc = pred.eq(&y)?.to_dtype(DType::F32)?.mean_all()?.to_scalar::<f32>()?;
            println!("step {step} loss={:.4} acc={acc:.3}", ce.to_scalar::<f32>()?);
        }
    }
    Ok(())
}
