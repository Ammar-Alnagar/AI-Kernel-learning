pub fn offset_2d(row: usize, col: usize, stride_row: usize, stride_col: usize) -> usize {
    // FILL IN
    todo!("fill offset_2d")
}

pub fn gather_2d_strided(buf: &[f32], rows: &[usize], cols: &[usize], stride_row: usize, stride_col: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows.len()];
    for i in 0..rows.len() {
        // FILL IN
        todo!("fill gather_2d_strided")
    }
    out
}

pub fn cast_array_i32(x: &[f32]) -> Vec<i32> {
    // FILL IN
    todo!("fill cast_array_i32")
}

pub fn is_power_of_two(x: usize) -> bool {
    // FILL IN
    todo!("fill is_power_of_two")
}
