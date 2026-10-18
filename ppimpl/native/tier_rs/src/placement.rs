//! `kvpool/placement.py`'s `BalancedPlacement.pieces` and `local` over the arrays Python built at attach (device
//! addresses already valid in this process).

pub struct Placement {
    pub slot_bytes: i64,
    pub kv_stage: Vec<i64>,
    pub kv_dev: Vec<i64>,
    pub kv_stride: Vec<i64>,
    pub kv_len: Vec<i64>,
    pub kv_host: Vec<i64>,
    pub st_stage: Vec<i64>,
    pub st_dev: Vec<i64>,
    pub st_stride: Vec<i64>,
    pub st_group: Vec<i64>,
    pub st_slot: Vec<i64>,
    pub st_off: Vec<i64>,
    pub st_len: Vec<i64>,
}

/// One list for every stage (`[[...]]`, `_shared`) or one per stage.
fn block(lists: &[Vec<i64>], stage: i64, i: usize) -> Result<i64, String> {
    let l = if lists.len() == 1 {
        &lists[0]
    } else {
        lists.get(stage as usize).ok_or_else(|| format!("no block list for stage {stage} ({} lists)", lists.len()))?
    };
    l.get(i).copied().ok_or_else(|| format!("block index {i} out of range ({} blocks)", l.len()))
}

pub type Pieces = (Vec<i64>, Vec<i64>, Vec<i64>);

impl Placement {
    /// Blocks [b_lo, b_hi): block j's pieces in host slot j - b_lo (`pieces(kv, st, range(b_hi - b_lo), [], b_lo,
    /// b_hi, b_lo)`), block-major, ranges in order.
    pub fn kv_pieces(&self, kv_blocks: &[Vec<i64>], b_lo: i64, b_hi: i64) -> Result<Pieces, String> {
        let (mut dev, mut off, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for j in b_lo..b_hi {
            for r in 0..self.kv_len.len() {
                let hb = block(kv_blocks, self.kv_stage[r], j as usize)?;
                dev.push(self.kv_dev[r] + hb * self.kv_stride[r]);
                off.push((j - b_lo) * self.slot_bytes + self.kv_host[r]);
                lens.push(self.kv_len[r]);
            }
        }
        Ok((dev, off, lens))
    }

    /// The state's pieces in state slots 0.. (`pieces(kv, st, [], range(len(state_slots)), 0, 0, 0)`).
    pub fn state_pieces(&self, state_blocks: &[Vec<i64>], n_slots: usize) -> Result<Pieces, String> {
        let (mut dev, mut off, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..self.st_len.len() {
            if self.st_slot[i] as usize >= n_slots {
                return Err(format!("state piece {i} in state slot {} of {n_slots}", self.st_slot[i]));
            }
            let sb = block(state_blocks, self.st_stage[i], self.st_group[i] as usize)?;
            dev.push(self.st_dev[i] + sb * self.st_stride[i]);
            off.push(self.st_slot[i] * self.slot_bytes + self.st_off[i]);
            lens.push(self.st_len[i]);
        }
        Ok((dev, off, lens))
    }

    /// Which pieces (`blocks` blocks, then the state if `state`) are in stage `stage`'s KV.
    pub fn local(&self, stage: i64, blocks: i64, state: bool) -> Vec<bool> {
        let mut out = Vec::new();
        for _ in 0..blocks.max(0) {
            out.extend(self.kv_stage.iter().map(|&s| s == stage));
        }
        if state {
            out.extend(self.st_stage.iter().map(|&s| s == stage));
        }
        out
    }
}
