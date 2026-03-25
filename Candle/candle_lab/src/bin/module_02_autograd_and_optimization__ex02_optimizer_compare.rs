use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, optim, Module, Optimizer, VarBuilder, VarMap};

fn run_adamw(x: &Tensor, y: &Tensor, dev: &Device) -> Result<f32> {
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, dev);
    let lin = candle_nn::linear(1, 1, vb.pp("lin"))?;
    let mut opt = optim::AdamW::new_lr(vm.all_vars(), 1e-2)?;
    let mut last = 0f32;
    for _ in 0..200 {
        let l = loss::mse(&lin.forward(x)?, y)?;
        last = l.to_scalar::<f32>()?;
        opt.backward_step(&l)?;
    }
    Ok(last)
}

fn main() -> Result<()> {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (256, 1), &dev)?;
    let y = x.affine(3.0, 2.0)?;
    let adamw = run_adamw(&x, &y, &dev)?;
    println!("final adamw loss={adamw:.6}");
    Ok(())
}
