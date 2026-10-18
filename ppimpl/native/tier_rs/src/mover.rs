//! `kvpool/mover.py` (layer 4): one GPU's copies between host slots and the stages' KV, copy engines only.
//!
//! Same rules as the Python `Mover`: two hops through a ring of `depth` chunks of `chunk` slots per direction in this
//! GPU's HBM (own-GPU pieces straight between host and KV in the PCIe batch, 5b9cec0), or direct; a transfer
//! without a gate holds its direction from its start mark to its last chunk; a gated one (a write yielding to
//! reads) takes it per chunk and first waits on the host for its previous chunk's first hop; per ring chunk two
//! events (first hop done, second hop done) recorded again at each use; a pooled event per op end and start mark,
//! two per batch with the trace. On CPU (tests) the copies are memmoves done at once; `Backend::Record` only
//! records each batch (plan comparison with the Python mover).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::cuda::{self, Ev};
use crate::util::{perf_ns, thread_cpu_ns};

pub const EVENT_POOL: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Read = 0,
    Write = 1,
}

// streams: read, write, read_nvl, write_nvl
pub const STREAM_NAMES: [&str; 4] = ["read", "write", "read_nvl", "write_nvl"];
const S_READ: usize = 0;
const S_WRITE: usize = 1;
const S_READ_NVL: usize = 2;
const S_WRITE_NVL: usize = 3;

/// per direction: (first hop's stream, second hop's stream); the ring sits between them
fn hops(d: Dir) -> (usize, usize) {
    match d {
        Dir::Read => (S_READ, S_READ_NVL),
        Dir::Write => (S_WRITE_NVL, S_WRITE),
    }
}

fn own_stream(d: Dir) -> usize {
    match d {
        Dir::Read => S_READ,
        Dir::Write => S_WRITE,
    }
}

/// Host slots and the pieces in them (`mover.Segment`): piece i's host side is `off[i]` bytes into the slots
/// counted slot bytes apart (slot k at host_ptrs[k]).
#[derive(Clone, Debug, Default)]
pub struct Segment {
    pub host_ptrs: Vec<i64>,
    pub dev: Vec<i64>,
    pub off: Vec<i64>,
    pub lens: Vec<i64>,
    pub local: Option<Vec<bool>>,
}

/// One traced batch: (stream, pieces, bytes, host ns before the call, after it, thread CPU ns in it, GPU start
/// event, end event).
pub struct BatchTrace {
    pub stream: usize,
    pub pieces: usize,
    pub bytes: i64,
    pub h0: i64,
    pub h1: i64,
    pub cpu: i64,
    pub e0: Ev,
    pub e1: Ev,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rec {
    pub stream: usize,
    pub src: Vec<i64>,
    pub dst: Vec<i64>,
    pub lens: Vec<i64>,
}

pub enum Backend {
    Cpu,
    Cuda { streams: [usize; 4] },
    Record(Mutex<Vec<Rec>>),
}

struct Ring {
    depth: usize,
    chunk: usize,
    base: [i64; 2],
}

pub struct Mover {
    pub sb: i64,
    pub backend: Backend,
    ring: Option<Ring>,
    _cpu_ring: Option<Vec<u8>>,
    // a direction's lock and its ring's next chunk
    dir: [Mutex<usize>; 2],
    pool: Mutex<Vec<Ev>>,
    all_events: Mutex<Vec<Ev>>,
    pub events_made: AtomicUsize,
    chunk_ev: [Vec<(Ev, Ev)>; 2],
    pub gate_wait_ns: Mutex<Vec<i64>>,
    pub record_ns: Mutex<Vec<i64>>,
    pub wait_ns: Mutex<Vec<i64>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn runs(src: &[i64], dst: &[i64], lens: &[i64], slot: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    // pieces in order merged into contiguous runs on both sides, within a slot (`mover._runs`)
    if lens.len() < 2 {
        return (src.to_vec(), dst.to_vec(), lens.to_vec());
    }
    let (mut s, mut d, mut l) = (vec![src[0]], vec![dst[0]], vec![lens[0]]);
    for i in 1..lens.len() {
        if src[i] != src[i - 1] + lens[i - 1] || dst[i] != dst[i - 1] + lens[i - 1] || slot[i] != slot[i - 1] {
            s.push(src[i]);
            d.push(dst[i]);
            l.push(lens[i]);
        } else {
            *l.last_mut().unwrap() += lens[i];
        }
    }
    (s, d, l)
}

pub type Gate<'a> = &'a (dyn Fn() + Sync);

impl Mover {
    /// `hops`: "two" or "direct". CUDA: `ring_ptr` is the device ring (2 x depth x chunk x slot bytes, Python's
    /// tensor); CPU / Record: a host ring is allocated here unless `ring_ptr` is given (Record: the Python mover's
    /// ring, so both plans carry the same addresses).
    pub fn new(backend: Backend, slot_bytes: i64, two_hops: bool, depth: usize, chunk: usize, ring_ptr: Option<i64>)
        -> Result<Mover, String> {
        let cudaish = matches!(backend, Backend::Cuda { .. });
        let mut pool = Vec::new();
        let mut chunk_ev: [Vec<(Ev, Ev)>; 2] = [Vec::new(), Vec::new()];
        let mut all = Vec::new();
        if let Backend::Cuda { streams } = &backend {
            // every event recorded once and waited for, as the Python pool's (torch makes it at its first record)
            for _ in 0..EVENT_POOL {
                let e = cuda::event_create(cuda::CU_EVENT_BLOCKING_SYNC)?;
                cuda::record(e, streams[S_READ])?;
                pool.push(e);
            }
            if two_hops {
                for d in 0..2 {
                    for _ in 0..depth {
                        let f = cuda::CU_EVENT_BLOCKING_SYNC | cuda::CU_EVENT_DISABLE_TIMING;
                        let pair = (cuda::event_create(f)?, cuda::event_create(f)?);
                        cuda::record(pair.0, streams[S_READ])?;
                        cuda::record(pair.1, streams[S_READ])?;
                        chunk_ev[d].push(pair);
                    }
                }
            }
            cuda::stream_sync(streams[S_READ])?;
            all.extend(pool.iter().copied());
            all.extend(chunk_ev.iter().flatten().flat_map(|p| [p.0, p.1]));
        }
        let mut cpu_ring = None;
        let ring = if two_hops {
            let half = depth as i64 * chunk as i64 * slot_bytes;
            let base = match ring_ptr {
                Some(p) => p,
                None => {
                    if cudaish {
                        return Err("two hops on CUDA need the device ring".into());
                    }
                    let v = vec![0u8; (2 * half + 4096) as usize];
                    let p = v.as_ptr() as i64;
                    cpu_ring = Some(v);
                    p
                }
            };
            Some(Ring { depth, chunk, base: [base, base + half] })
        } else {
            None
        };
        Ok(Mover {
            sb: slot_bytes,
            backend,
            ring,
            _cpu_ring: cpu_ring,
            dir: [Mutex::new(0), Mutex::new(0)],
            events_made: AtomicUsize::new(if cudaish { EVENT_POOL } else { 0 }),
            pool: Mutex::new(pool),
            all_events: Mutex::new(all),
            chunk_ev,
            gate_wait_ns: Mutex::new(Vec::new()),
            record_ns: Mutex::new(Vec::new()),
            wait_ns: Mutex::new(Vec::new()),
        })
    }

    pub fn cuda(&self) -> bool {
        matches!(self.backend, Backend::Cuda { .. })
    }

    fn stream(&self, i: usize) -> usize {
        match &self.backend {
            Backend::Cuda { streams } => streams[i],
            _ => 0,
        }
    }

    fn drv_record(&self, e: Ev, stream: usize) -> Result<(), String> {
        let t = perf_ns();
        let r = cuda::record(e, self.stream(stream));
        lock(&self.record_ns).push(perf_ns() - t);
        r
    }

    fn drv_wait(&self, stream: usize, e: Ev) -> Result<(), String> {
        let t = perf_ns();
        let r = cuda::wait(self.stream(stream), e);
        lock(&self.wait_ns).push(perf_ns() - t);
        r
    }

    /// A pooled event (blocking sync, timing), appended to `owned` (released once the op is done).
    pub fn event(&self, owned: &mut Vec<Ev>) -> Result<Ev, String> {
        let e = lock(&self.pool).pop();
        let e = match e {
            Some(e) => e,
            None => {
                self.events_made.fetch_add(1, Ordering::Relaxed);
                let e = cuda::event_create(cuda::CU_EVENT_BLOCKING_SYNC)?;
                cuda::record(e, self.stream(S_READ))?;
                lock(&self.all_events).push(e);
                e
            }
        };
        owned.push(e);
        Ok(e)
    }

    pub fn release(&self, owned: Vec<Ev>) {
        if !owned.is_empty() {
            lock(&self.pool).extend(owned);
        }
    }

    /// A timing event on the stream a transfer's first copy goes on (CUDA; None otherwise).
    fn mark(&self, d: Dir, owned: &mut Vec<Ev>) -> Result<Option<Ev>, String> {
        if !self.cuda() {
            return Ok(None);
        }
        let e = self.event(owned)?;
        self.drv_record(e, if self.ring.is_some() { hops(d).0 } else { own_stream(d) })?;
        Ok(Some(e))
    }

    fn end(&self, stream: usize, owned: &mut Vec<Ev>) -> Result<Option<Ev>, String> {
        if !self.cuda() {
            return Ok(None);
        }
        let e = self.event(owned)?;
        self.drv_record(e, stream)?;
        Ok(Some(e))
    }

    /// "read" host -> HBM or "write" HBM -> host of every segment's pieces (`Mover.transfer_async`). Returns (the
    /// transfer's end event, bytes of HBM pieces, the start mark if `want_mark`).
    pub fn transfer(&self, d: Dir, segs: &[Segment], tr: Option<&mut Vec<BatchTrace>>, owned: &mut Vec<Ev>,
                    gate: Option<Gate>, want_mark: bool) -> Result<(Option<Ev>, i64, Option<Ev>), String> {
        let mut mark = None;
        if gate.is_none() {
            let mut g = lock(&self.dir[d as usize]);
            if want_mark {
                mark = self.mark(d, owned)?;
            }
            let (e, n) = self.inner(d, segs, tr, owned, None, Some(&mut g))?;
            Ok((e, n, mark))
        } else {
            if want_mark {
                let _g = lock(&self.dir[d as usize]);
                mark = self.mark(d, owned)?;
            }
            let (e, n) = self.inner(d, segs, tr, owned, gate, None)?;
            Ok((e, n, mark))
        }
    }

    fn inner(&self, d: Dir, segs: &[Segment], mut tr: Option<&mut Vec<BatchTrace>>, owned: &mut Vec<Ev>,
             gate: Option<Gate>, mut held: Option<&mut MutexGuard<'_, usize>>) -> Result<(Option<Ev>, i64), String> {
        let sb = self.sb;
        let mut first = vec![0i64];
        for s in segs {
            first.push(first.last().unwrap() + s.host_ptrs.len() as i64);
        }
        let n = *first.last().unwrap();
        let hosts: Vec<i64> = segs.iter().flat_map(|s| s.host_ptrs.iter().copied()).collect();
        let (mut gslot, mut inslot, mut dev, mut lens, mut local) = (vec![], vec![], vec![], vec![], vec![]);
        for (k, s) in segs.iter().enumerate() {
            if s.dev.len() != s.lens.len() || s.off.len() != s.lens.len() {
                return Err("segment arrays of different lengths".into());
            }
            for i in 0..s.lens.len() {
                let g = s.off[i].div_euclid(sb) + first[k];
                if g < first[k] || g >= first[k + 1] {
                    return Err(format!("piece at host offset {} is outside its segment's {} slots", s.off[i],
                                       s.host_ptrs.len()));
                }
                gslot.push(g);
                inslot.push(s.off[i].rem_euclid(sb));
                dev.push(s.dev[i]);
                lens.push(s.lens[i]);
                local.push(s.local.as_ref().map_or(false, |l| l[i]));
            }
        }
        let nbytes: i64 = lens.iter().sum();
        let hpiece: Vec<i64> = (0..lens.len()).map(|i| hosts[gslot[i] as usize] + inslot[i]).collect();
        let ring = match &self.ring {
            None => {
                if lens.is_empty() {
                    return Ok((None, 0));
                }
                let (src, dst) = if d == Dir::Read { (&hpiece, &dev) } else { (&dev, &hpiece) };
                let mut tmp;
                let _g = if held.is_none() {
                    tmp = Some(lock(&self.dir[d as usize]));
                    tmp.as_mut()
                } else {
                    None
                };
                self.batch(own_stream(d), src, dst, &lens, None, None, tr.as_deref_mut(), owned)?;
                return Ok((self.end(own_stream(d), owned)?, nbytes));
            }
            Some(r) => r,
        };
        let c = ring.chunk as i64;
        let prel: Vec<i64> = (0..lens.len()).map(|i| (gslot[i] % c) * sb + inslot[i]).collect();
        let (s1, s2) = hops(d);
        let mut end = None;
        let mut prev: Option<Ev> = None; // the previous chunk's first hop done (gated transfers)
        let mut a = 0i64;
        let mut lo = 0usize;
        while a < n {
            let b = (a + c).min(n);
            if let Some(g) = gate {
                if let Some(p) = prev {
                    let t = perf_ns();
                    cuda::synchronize(p)?;
                    lock(&self.gate_wait_ns).push(perf_ns() - t);
                }
                g();
            }
            let hi = lo + gslot[lo..].partition_point(|&x| x < a + c);
            let mut tmp;
            let st: &mut usize = match held.as_deref_mut() {
                Some(g) => &mut **g,
                None => {
                    tmp = lock(&self.dir[d as usize]);
                    &mut *tmp
                }
            };
            let pos = *st;
            *st = (pos + 1) % ring.depth;
            let rbase = ring.base[d as usize] + pos as i64 * c * sb;
            let (ev1, ev2) = if self.cuda() {
                let p = self.chunk_ev[d as usize][pos];
                (Some(p.0), Some(p.1))
            } else {
                (None, None)
            };
            let (mut rp, mut dp, mut lp, mut hp, mut gp) = (vec![], vec![], vec![], vec![], vec![]);
            let (mut hl, mut dl, mut ll) = (vec![], vec![], vec![]);
            for i in lo..hi {
                if local[i] {
                    hl.push(hpiece[i]);
                    dl.push(dev[i]);
                    ll.push(lens[i]);
                } else {
                    rp.push(rbase + prel[i]);
                    dp.push(dev[i]);
                    lp.push(lens[i]);
                    hp.push(hpiece[i]);
                    gp.push(gslot[i]);
                }
            }
            let (hr, rr, lr) = runs(&hp, &rp, &lp, &gp); // host side, runs per slot
            let cat = |x: &[i64], y: &[i64]| -> Vec<i64> { x.iter().chain(y.iter()).copied().collect() };
            if d == Dir::Read {
                // host -> ring and own pieces (PCIe), then ring -> the peers (NVLink)
                self.batch(s1, &cat(&hr, &hl), &cat(&rr, &dl), &cat(&lr, &ll), ev2, ev1, tr.as_deref_mut(),
                           owned)?;
                self.batch(s2, &rp, &dp, &lp, ev1, ev2, tr.as_deref_mut(), owned)?;
            } else {
                // the peers -> ring, then ring and own pieces -> host
                self.batch(s1, &dp, &rp, &lp, ev2, ev1, tr.as_deref_mut(), owned)?;
                if gate.is_some() {
                    prev = ev1;
                }
                self.batch(s2, &cat(&rr, &dl), &cat(&hr, &hl), &cat(&lr, &ll), ev1, ev2, tr.as_deref_mut(),
                           owned)?;
            }
            if b == n {
                end = self.end(s2, owned)?;
            }
            lo = hi;
            a += c;
        }
        Ok((end, nbytes))
    }

    /// One batch on `stream` after the event `wait`; records `done` after it (`Mover._batch`).
    #[allow(clippy::too_many_arguments)]
    fn batch(&self, stream: usize, src: &[i64], dst: &[i64], lens: &[i64], wait: Option<Ev>, done: Option<Ev>,
             tr: Option<&mut Vec<BatchTrace>>, owned: &mut Vec<Ev>) -> Result<(), String> {
        match &self.backend {
            Backend::Cpu => {
                for i in 0..lens.len() {
                    unsafe {
                        std::ptr::copy(src[i] as *const u8, dst[i] as *mut u8, lens[i] as usize);
                    }
                }
                Ok(())
            }
            Backend::Record(r) => {
                lock(r).push(Rec { stream, src: src.to_vec(), dst: dst.to_vec(), lens: lens.to_vec() });
                Ok(())
            }
            Backend::Cuda { streams } => {
                let st = streams[stream];
                if let Some(w) = wait {
                    self.drv_wait(stream, w)?;
                }
                if lens.is_empty() {
                    if let Some(e) = done {
                        self.drv_record(e, stream)?;
                    }
                    return Ok(());
                }
                let mut t = None;
                if tr.is_some() {
                    let e0 = self.event(owned)?;
                    self.drv_record(e0, stream)?;
                    t = Some((e0, perf_ns(), thread_cpu_ns()));
                }
                cuda::batch(st, src, dst, lens)?;
                if let (Some(tr), Some((e0, h0, c0))) = (tr, t) {
                    let (h1, c1) = (perf_ns(), thread_cpu_ns());
                    let e1 = self.event(owned)?;
                    self.drv_record(e1, stream)?;
                    tr.push(BatchTrace { stream, pieces: lens.len(), bytes: lens.iter().sum(), h0, h1, cpu: c1 - c0,
                                         e0, e1 });
                }
                if let Some(e) = done {
                    self.drv_record(e, stream)?;
                }
                Ok(())
            }
        }
    }

    pub fn set_ring_next(&self, d: Dir, pos: usize) {
        *lock(&self.dir[d as usize]) = pos;
    }

    pub fn take_records(&self) -> Vec<Rec> {
        match &self.backend {
            Backend::Record(r) => std::mem::take(&mut *lock(r)),
            _ => Vec::new(),
        }
    }

    pub fn destroy_events(&self) {
        if self.cuda() {
            for e in lock(&self.all_events).drain(..) {
                cuda::event_destroy(e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_merge_within_a_slot_only() {
        let (s, d, l) = runs(&[0, 10, 20, 50], &[100, 110, 120, 130], &[10, 10, 10, 5], &[0, 0, 1, 1]);
        assert_eq!((s, d, l), (vec![0, 20, 50], vec![100, 120, 130], vec![20, 10, 5]));
    }

    fn segs(host: &mut [u8], dev: &mut [u8], local: bool) -> Vec<Segment> {
        // as tests/test_kvpool_mover.py: 5 KV slots of 3 pieces, 2 state slots of pieces split at the slot boundary
        let sb = 4096i64;
        let hb = host.as_ptr() as i64;
        let hb = hb + (-hb).rem_euclid(sb);
        let base = dev.as_ptr() as i64;
        let kv_len: Vec<i64> = (0..5).flat_map(|_| [1024, 768, 1024]).collect();
        let kv_off: Vec<i64> = (0..5).flat_map(|s| [s * sb, s * sb + 1024, s * sb + 1792]).collect();
        let pos: Vec<i64> = (0..18).map(|i| base + ((i * 7) % 64) * 4096).collect();
        let kv_loc = local.then(|| (0..15).map(|i| i % 3 == 1).collect());
        let st_loc = local.then(|| vec![false, true, false]);
        vec![
            Segment { host_ptrs: [6, 2, 0, 4, 8].iter().map(|s| hb + s * sb).collect(), dev: pos[..15].to_vec(),
                      off: kv_off, lens: kv_len, local: kv_loc },
            Segment { host_ptrs: [1, 9].iter().map(|s| hb + s * sb).collect(), dev: pos[15..].to_vec(),
                      off: vec![0, 3000, sb], lens: vec![3000, 1096, 2000], local: st_loc },
        ]
    }

    #[test]
    fn write_then_read_is_byte_equal() {
        for (two, depth, chunk) in [(false, 2, 4), (true, 2, 4), (true, 1, 1), (true, 2, 3)] {
            for local in [false, true] {
                let mut dev: Vec<u8> = (0..64 * 4096).map(|i| (i * 31 % 251) as u8).collect();
                let mut host = vec![0u8; 12 * 4096];
                let s = segs(&mut host, &mut dev, local);
                let want = dev.clone();
                let m = Mover::new(Backend::Cpu, 4096, two, depth, chunk, None).unwrap();
                let mut owned = vec![];
                let (e, n, _) = m.transfer(Dir::Write, &s, None, &mut owned, None, false).unwrap();
                assert!(e.is_none());
                assert_eq!(n, s.iter().map(|x| x.lens.iter().sum::<i64>()).sum::<i64>());
                dev.iter_mut().for_each(|x| *x = 0);
                let gate = || {};
                m.transfer(Dir::Read, &s, None, &mut owned, Some(&gate), true).unwrap();
                let base = dev.as_ptr() as i64;
                for sg in &s {
                    for i in 0..sg.lens.len() {
                        let a = (sg.dev[i] - base) as usize;
                        assert_eq!(dev[a..a + sg.lens[i] as usize], want[a..a + sg.lens[i] as usize]);
                    }
                }
            }
        }
    }
}
