pub fn linear_pid_to_2d(pid: usize, grid_n: usize) -> (usize, usize) {
    // FILL IN
    todo!("fill linear_pid_to_2d")
}

pub fn grouped_pid_to_2d(pid: usize, grid_m: usize, grid_n: usize, group_m: usize) -> (usize, usize) {
    let blocks_per_group = group_m * grid_n;
    let group_id = pid / blocks_per_group;
    let first_m = group_id * group_m;
    // FILL IN
    todo!("fill grouped_pid_to_2d")
}

pub fn tile_bounds(pid_m: usize, pid_n: usize, bm: usize, bn: usize, m: usize, n: usize) -> ((usize, usize), (usize, usize)) {
    // FILL IN
    todo!("fill tile_bounds")
}
