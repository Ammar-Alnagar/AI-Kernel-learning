use candle_core::{Result, Tensor};

struct KvCache {
    k: Tensor,
    v: Tensor,
    cur_len: usize,
}

impl KvCache {
    fn append(&mut self, k_t: &Tensor, v_t: &Tensor) -> Result<()> {
        // TODO: update internal tensors at cur_len
        self.cur_len += 1;
        Ok(())
    }

    fn prefix(&self) -> Result<(Tensor, Tensor)> {
        // TODO: return views up to cur_len
        let k = _____;
        let v = _____;
        Ok((k, v))
    }
}

fn main() {}
