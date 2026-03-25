pub fn persistent_tile_ids(block_id: usize, num_blocks: usize, num_tiles: usize) -> Vec<usize> {
    let mut ids = Vec::new();
    let mut tile = block_id;
    while tile < num_tiles {
        ids.push(tile);
        tile += num_blocks;
    }
    ids
}

pub fn persistent_vector_add(x: &[f32], y: &[f32], tile_size: usize, block_id: usize, num_blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    let num_tiles = (x.len() + tile_size - 1) / tile_size;
    let mut tile = block_id;
    while tile < num_tiles {
        let start = tile * tile_size;
        let end = usize::min(start + tile_size, x.len());
        for i in start..end {
            out[i] = x[i] + y[i];
        }
        tile += num_blocks;
    }
    out
}
