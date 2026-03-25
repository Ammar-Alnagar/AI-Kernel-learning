use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::Cpu;

    let a = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev)?;
    let b = Tensor::ones((2, 2), DType::F32, &dev)?;

    let c = (&a + &b)?;
    let d = a.matmul(&c)?;

    let flat = d.reshape((4,))?;
    let row = flat.reshape((1, 4))?;

    println!("d shape: {:?}", d.shape());
    println!("d values: {:?}", d.to_vec2::<f32>()?);
    println!("row values: {:?}", row.to_vec2::<f32>()?);
    Ok(())
}
