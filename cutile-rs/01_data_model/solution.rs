pub fn offset_2d(row: usize, col: usize, stride_row: usize, stride_col: usize) -> usize {
    row * stride_row + col * stride_col
}

pub fn gather_2d_strided(buf: &[f32], rows: &[usize], cols: &[usize], stride_row: usize, stride_col: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows.len()];
    for i in 0..rows.len() {
        let idx = offset_2d(rows[i], cols[i], stride_row, stride_col);
        out[i] = buf[idx];
    }
    out
}

pub fn cast_array_i32(x: &[f32]) -> Vec<i32> {
    x.iter().map(|v| *v as i32).collect()
}

pub fn is_power_of_two(x: usize) -> bool {
    x > 0 && (x & (x - 1)) == 0
}
