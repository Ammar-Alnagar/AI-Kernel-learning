use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, optim::AdamW, Module, VarBuilder, VarMap};

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let n = 256;

    let x = Tensor::randn(0f32, 1f32, (n, 1), &dev)?;
    let y = x.affine(3.0, 2.0)?;

    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    let lin = candle_nn::linear(1, 1, vb.pp("lin"))?;
    let mut opt = AdamW::new_lr(vm.all_vars(), 1e-2)?;

    for step in 0..200 {
        let pred = lin.forward(&x)?;
        let l = loss::mse(&pred, &y)?;
        opt.backward_step(&l)?;
        if step % 50 == 0 {
            println!("step {step} loss={:.6}", l.to_scalar::<f32>()?);
        }
    }

    Ok(())
}
