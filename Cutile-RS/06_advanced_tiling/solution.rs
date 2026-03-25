pub fn linear_pid_to_2d(pid: usize, grid_n: usize) -> (usize, usize) {
    (pid / grid_n, pid % grid_n)
}

pub fn grouped_pid_to_2d(pid: usize, grid_m: usize, grid_n: usize, group_m: usize) -> (usize, usize) {
    let blocks_per_group = group_m * grid_n;
    let group_id = pid / blocks_per_group;
    let first_m = group_id * group_m;
    let group_size_m = usize::min(group_m, grid_m - first_m);
    let pid_in_group = pid % blocks_per_group;
    (first_m + (pid_in_group % group_size_m), pid_in_group / group_size_m)
}

pub fn tile_bounds(pid_m: usize, pid_n: usize, bm: usize, bn: usize, m: usize, n: usize) -> ((usize, usize), (usize, usize)) {
    let row_start = pid_m * bm;
    let col_start = pid_n * bn;
    ((row_start, usize::min(row_start + bm, m)), (col_start, usize::min(col_start + bn, n)))
}
