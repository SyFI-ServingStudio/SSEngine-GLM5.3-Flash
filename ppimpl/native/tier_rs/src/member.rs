//! `kvpool/member.py`'s `PoolMember` threads, one-to-one: recv, submit per direction, done per direction, trace,
//! SSD per direction, SSD copy (and the io_uring's thread in `uring.rs`). Same queues, same order of work, same end
//! chain (recv -> submit -> SSD -> copy -> done -> trace), errors passed down it as `WorkerError`. No Python.

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::cuda::{self, Ctx, Ev};
use crate::mover::{BatchTrace, Dir, Mover, Segment, STREAM_NAMES};
use crate::placement::Placement;
use crate::uring::UringFile;
use crate::util::{perf_ns, time_ns};
use crate::wire::{self, Done, Msg, Op};

const ANCHOR_S: f64 = 1.0; // a thread's GPU-to-host clock anchor is taken again after this long
const POLL_MS: i64 = 200;

pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Config {
    pub stage: i64,
    pub real: bool,
    pub slot_bytes: i64,
    pub dram_base: Option<i64>,
    pub staging_base: Option<i64>,
    pub trace_every: i64, // 0: no trace
    pub run_dir: String,
    pub dram_read_gb_s: f64,
    pub ssd_read_gb_s: f64,
}

pub enum Wait {
    None,
    Ev(Ev),
    Due(Instant), // fake member
}

pub struct DoneItem {
    op_id: i64,
    wait: Wait,
    start_ns: i64,
    nbytes: i64,
    submit_ns: i64,
    mark: Option<Ev>,
    recv_ns: Option<i64>,
    trace: Option<(String, i64, Vec<BatchTrace>)>,
    owned: Vec<Ev>,
}

enum Sub {
    Op(Op),
    End(Option<String>),
}

enum SsdItem {
    Op(Op, i64),
    End(Option<String>),
}

enum CopyItem {
    Op(Op, i64, usize, Option<String>),
    End(Option<String>),
}

enum QItem {
    Done(DoneItem),
    Err(String),
    End,
}

enum TItem {
    Op(i64, i64, Option<Ev>, Option<(String, i64, Vec<BatchTrace>)>, Vec<Ev>),
    End,
}

pub enum Report {
    Done(Done),
    Error(String),
}

/// reads to HBM queued or in flight, counted in and out; a write chunk waits for the reads counted in when it is due
struct ReadGate {
    m: Mutex<(i64, i64, HashSet<i64>)>,
    cv: Condvar,
    waits: AtomicU64,
    wait_ms: Mutex<f64>,
}

impl ReadGate {
    fn queued(&self, op_id: i64) {
        let mut g = lock(&self.m);
        g.0 += 1;
        g.2.insert(op_id);
    }

    fn done(&self, op_id: i64) {
        let mut g = lock(&self.m);
        if g.2.remove(&op_id) {
            g.1 += 1;
            self.cv.notify_all();
        }
    }

    /// A write's next ring chunk: wait until the reads queued or in flight now have completed.
    fn yield_to_reads(&self, stop: &AtomicBool) {
        let mut g = lock(&self.m);
        let target = g.0;
        if g.1 >= target {
            return;
        }
        let t0 = Instant::now();
        while !(g.1 >= target || stop.load(Ordering::Relaxed)) {
            g = self.cv.wait_timeout(g, Duration::from_millis(POLL_MS as u64)).unwrap_or_else(|e| e.into_inner()).0;
        }
        self.waits.fetch_add(1, Ordering::Relaxed);
        *lock(&self.wait_ms) += t0.elapsed().as_secs_f64() * 1e3;
    }
}

pub struct Chan<T> {
    pub tx: Sender<T>,
    pub rx: Receiver<T>,
}

fn chan<T>() -> Chan<T> {
    let (tx, rx) = unbounded();
    Chan { tx, rx }
}

pub struct Member {
    pub cfg: Config,
    pub placement: Option<Placement>,
    pub mover: Option<Mover>,
    pub ssd: Option<UringFile>,
    pub ctx: Option<Ctx>,
    zctx: zmq::Context,
    pub stop: AtomicBool,
    gate: ReadGate,
    ssd_inflight: Mutex<i64>,
    ssd_cv: Condvar,
    due: Mutex<[Option<Instant>; 2]>,
    trace_on: AtomicBool,
    trace_log: Mutex<Option<File>>,
    sub: [Chan<Sub>; 2],
    ssd_q: [Chan<SsdItem>; 2],
    copy_q: Chan<CopyItem>,
    q: [Chan<QItem>; 2],
    trace_q: Chan<TItem>,
    pub local: Chan<Report>,
    pub ended: AtomicUsize, // chain ends sent by recv (once)
    /// (tier socket to bind, report socket to connect; None: completions to `local`, tests), set by `start`
    pub addrs: Mutex<(Option<String>, Option<String>)>,
}

fn dir_of(op: &Op) -> Dir {
    if op.op == "read" || op.op == "fetch" {
        Dir::Read
    } else {
        Dir::Write
    }
}

/// Tracks one op's file I/O: its callback runs once all requests completed (`left` holds one guard count released
/// after the last submission, as the Python callbacks attach after every slot is queued).
struct Tracker {
    left: AtomicUsize,
    first_err: Mutex<Option<(usize, String)>>,
    failed: AtomicBool,
    abandoned: AtomicBool,
}

impl Member {
    pub fn new(cfg: Config, placement: Option<Placement>, mover: Option<Mover>, ssd: Option<UringFile>,
               ctx: Option<Ctx>) -> Member {
        let trace = mover.as_ref().is_some_and(|m| m.cuda()) && cfg.trace_every > 0;
        Member {
            cfg,
            placement,
            mover,
            ssd,
            ctx,
            zctx: zmq::Context::new(),
            stop: AtomicBool::new(false),
            gate: ReadGate { m: Mutex::new((0, 0, HashSet::new())), cv: Condvar::new(), waits: AtomicU64::new(0),
                             wait_ms: Mutex::new(0.0) },
            ssd_inflight: Mutex::new(0),
            ssd_cv: Condvar::new(),
            due: Mutex::new([None, None]),
            trace_on: AtomicBool::new(trace),
            trace_log: Mutex::new(None),
            sub: [chan(), chan()],
            ssd_q: [chan(), chan()],
            copy_q: chan(),
            q: [chan(), chan()],
            trace_q: chan(),
            local: chan(),
            ended: AtomicUsize::new(0),
            addrs: Mutex::new((None, None)),
        }
    }

    pub fn yields(&self) -> (u64, f64) {
        (self.gate.waits.load(Ordering::Relaxed), *lock(&self.gate.wait_ms))
    }

    fn worker_error(&self, what: &str, e: &str) -> String {
        format!("{what}:\n{e}")
    }

    fn current(&self) -> Result<(), String> {
        match &self.ctx {
            Some(c) => c.make_current(),
            None => Ok(()),
        }
    }

    // --- threads --------------------------------------------------------------------------------------------------

    pub fn recv_loop(self: &Arc<Self>) {
        let mut err = None;
        let r: Result<(), String> = (|| {
            self.current()?;
            let sock = self.zctx.socket(zmq::PULL).map_err(|e| e.to_string())?;
            let addr = lock(&self.addrs).0.clone().ok_or("no tier address")?;
            sock.bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
            let out = (|| {
                while !self.stop.load(Ordering::Relaxed) {
                    if sock.poll(zmq::POLLIN, POLL_MS).map_err(|e| e.to_string())? == 0 {
                        continue;
                    }
                    let buf = sock.recv_bytes(0).map_err(|e| e.to_string())?;
                    if !self.handle(&buf)? {
                        break;
                    }
                }
                Ok(())
            })();
            let _ = sock.set_linger(0);
            out
        })();
        if let Err(e) = r {
            err = Some(self.worker_error("tier thread", &e));
        }
        self.end_chain(err);
    }

    /// The end of the receive side: the read submit thread gets the error (if any), the write one the end.
    pub fn end_chain(&self, err: Option<String>) {
        if self.ended.fetch_add(1, Ordering::SeqCst) == 0 {
            let _ = self.sub[0].tx.send(Sub::End(err));
            let _ = self.sub[1].tx.send(Sub::End(None));
        }
    }

    /// One message off the tier socket (the receive thread; `Engine.dispatch` in tests). False: Shutdown.
    pub fn handle(&self, buf: &[u8]) -> Result<bool, String> {
        let mut ops = match wire::decode(buf)? {
            Msg::Shutdown => return Ok(false),
            Msg::Batch(ops) => ops,
        };
        let now = time_ns();
        for op in ops.iter_mut() {
            op.recv_ns = Some(now);
        }
        ops.sort_by_key(|o| o.op != "read"); // reads first (stable)
        for op in ops {
            let dram = self.cfg.real && op.tier == "dram";
            if dram && op.op == "read" {
                self.gate.queued(op.op_id);
            }
            if dram {
                let _ = self.sub[dir_of(&op) as usize].tx.send(Sub::Op(op));
            } else {
                self.submit(op)?; // SSD: to its thread; fake: timed here
            }
        }
        Ok(true)
    }

    pub fn submit_loop(self: &Arc<Self>, d: Dir) {
        let mut err = None;
        let r: Result<(), String> = (|| {
            self.current()?;
            loop {
                match self.sub[d as usize].rx.recv() {
                    Ok(Sub::Op(op)) => self.submit(op)?,
                    Ok(Sub::End(e)) => {
                        err = e;
                        return Ok(());
                    }
                    Err(_) => return Ok(()),
                }
            }
        })();
        if let Err(e) = r {
            let name = if d == Dir::Read { "read" } else { "write" };
            err = Some(self.worker_error(&format!("tier {name} submit thread"), &e));
        }
        if self.ssd.is_some() {
            let _ = self.ssd_q[d as usize].tx.send(SsdItem::End(err));
        } else {
            let _ = self.q[d as usize].tx.send(err.map_or(QItem::End, QItem::Err));
        }
    }

    pub fn ssd_loop(self: &Arc<Self>, d: Dir) {
        let mut err = None;
        let r: Result<(), String> = (|| {
            self.current()?;
            loop {
                match self.ssd_q[d as usize].rx.recv() {
                    Ok(SsdItem::Op(op, start)) => self.ssd_op(op, start)?,
                    Ok(SsdItem::End(e)) => {
                        err = e;
                        return Ok(());
                    }
                    Err(_) => return Ok(()),
                }
            }
        })();
        if let Err(e) = r {
            err = Some(self.worker_error("tier SSD thread", &e));
        }
        {
            // the writes in flight report before the end does
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut g = lock(&self.ssd_inflight);
            while *g != 0 && Instant::now() < deadline {
                g = self.ssd_cv.wait_timeout(g, Duration::from_millis(POLL_MS as u64))
                    .unwrap_or_else(|e| e.into_inner()).0;
            }
        }
        if d == Dir::Read {
            let _ = self.copy_q.tx.send(CopyItem::End(err));
        } else {
            let _ = self.q[1].tx.send(err.map_or(QItem::End, QItem::Err));
        }
    }

    pub fn copy_loop(self: &Arc<Self>) {
        let mut err = None;
        let r: Result<(), String> = (|| {
            self.current()?;
            loop {
                match self.copy_q.rx.recv() {
                    Ok(CopyItem::Op(op, start, n, e)) => self.ssd_read_copy(op, start, n, e)?,
                    Ok(CopyItem::End(e)) => {
                        err = e;
                        return Ok(());
                    }
                    Err(_) => return Ok(()),
                }
            }
        })();
        if let Err(e) = r {
            err = Some(self.worker_error("tier SSD copy thread", &e));
        }
        let _ = self.q[0].tx.send(err.map_or(QItem::End, QItem::Err));
    }

    fn reporter(&self) -> Result<Option<zmq::Socket>, String> {
        match &lock(&self.addrs).1 {
            None => Ok(None),
            Some(a) => {
                let s = self.zctx.socket(zmq::PUSH).map_err(|e| e.to_string())?;
                s.set_linger(5000).map_err(|e| e.to_string())?;
                s.connect(a).map_err(|e| format!("connect {a}: {e}"))?;
                Ok(Some(s))
            }
        }
    }

    fn send(&self, sock: &Option<zmq::Socket>, r: Report) -> Result<(), String> {
        match sock {
            Some(s) => {
                let buf = match &r {
                    Report::Done(d) => wire::encode_done(d),
                    Report::Error(e) => wire::encode_error(self.cfg.stage, e),
                };
                s.send(buf, 0).map_err(|e| e.to_string())
            }
            None => self.local.tx.send(r).map_err(|e| e.to_string()),
        }
    }

    pub fn done_loop(self: &Arc<Self>, d: Dir) {
        let push = self.reporter();
        let r: Result<(), String> = (|| {
            let push = push.as_ref().map_err(|e| e.clone())?;
            self.current()?;
            let mut anchor = Anchor::default();
            loop {
                let it = match self.q[d as usize].rx.recv() {
                    Ok(QItem::Done(it)) => it,
                    Ok(QItem::Err(e)) => {
                        self.send(push, Report::Error(e))?;
                        return Ok(());
                    }
                    Ok(QItem::End) | Err(_) => return Ok(()),
                };
                let (mut gpu_ns, mut mark_ns) = (None, None);
                match it.wait {
                    Wait::Due(t) => {
                        let now = Instant::now();
                        if t > now {
                            std::thread::sleep(t - now);
                        }
                    }
                    Wait::Ev(e) => {
                        cuda::synchronize(e)?;
                        if let Some(m) = it.mark {
                            gpu_ns = Some((cuda::elapsed_ms(m, e)? as f64 * 1e6) as i64);
                            mark_ns = Some(anchor.host_ns(m)?);
                        }
                    }
                    Wait::None => {}
                }
                self.send(push, Report::Done(Done {
                    stage: self.cfg.stage, op_id: it.op_id, start_ns: it.start_ns, end_ns: time_ns(),
                    nbytes: it.nbytes, submit_ns: it.submit_ns, gpu_ns, mark_ns, recv_ns: it.recv_ns,
                }))?;
                if d == Dir::Read {
                    self.gate.done(it.op_id);
                }
                if it.trace.is_some() || !it.owned.is_empty() {
                    // its copies completed: the trace thread reads its trace and releases its events
                    let _ = self.trace_q.tx.send(TItem::Op(it.op_id, it.start_ns, it.mark, it.trace, it.owned));
                }
            }
        })();
        if let Err(e) = r {
            let name = if d == Dir::Read { "read" } else { "write" };
            let msg = self.worker_error(&format!("tier {name} done thread"), &e);
            if let Ok(p) = &push {
                let _ = self.send(p, Report::Error(msg));
            }
        }
        let _ = self.trace_q.tx.send(TItem::End);
        anchor_drop();
    }

    /// Done ops' per-batch traces and pooled events (back to the mover), after their TierDone. Ends after both done
    /// threads did.
    pub fn trace_loop(self: &Arc<Self>) {
        let _ = self.current();
        let mut ended = 0;
        while ended < 2 {
            match self.trace_q.rx.recv() {
                Ok(TItem::End) | Err(_) => ended += 1,
                Ok(TItem::Op(op_id, start_ns, mark, tr, owned)) => {
                    if let (Some((kind, t0, tr)), true) = (tr, self.trace_on.load(Ordering::Relaxed)) {
                        if let Err(e) = self.write_trace(op_id, start_ns, mark, &kind, t0, &tr) {
                            println!("stage {} tier trace off:\n{e}", self.cfg.stage);
                            self.trace_on.store(false, Ordering::Relaxed);
                        }
                    }
                    if !owned.is_empty() {
                        if let Some(m) = &self.mover {
                            m.release(owned);
                        }
                    }
                }
            }
        }
    }

    fn write_trace(&self, op_id: i64, start_ns: i64, mark: Option<Ev>, kind: &str, t0: i64, tr: &[BatchTrace])
        -> Result<(), String> {
        let Some(mark) = mark else { return Ok(()) };
        let us = |ns: i64| ((ns as f64 / 1e3) * 10.0).round() / 10.0;
        let ms = |e: Ev| -> Result<f64, String> { Ok(((cuda::elapsed_ms(mark, e)? as f64 * 1e3) * 10.0).round() / 10.0) };
        let mut batches = Vec::new();
        for b in tr {
            batches.push(serde_json::json!({
                "stream": STREAM_NAMES[b.stream], "pieces": b.pieces, "bytes": b.bytes, "host_at_us": us(b.h0 - t0),
                "call_us": us(b.h1 - b.h0), "cpu_us": us(b.cpu), "gpu_start_us": ms(b.e0)?, "gpu_end_us": ms(b.e1)?,
            }));
        }
        let line = serde_json::json!({"op": op_id, "kind": kind, "start_ns": start_ns, "every": self.cfg.trace_every,
                                      "batches": batches});
        let mut f = lock(&self.trace_log);
        if f.is_none() {
            let p = format!("{}/tier_trace_s{}.jsonl", self.cfg.run_dir, self.cfg.stage);
            *f = Some(std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| format!("{p}: {e}"))?);
        }
        writeln!(f.as_mut().unwrap(), "{line}").map_err(|e| e.to_string())
    }

    // --- one op -----------------------------------------------------------------------------------------------------

    pub fn submit(&self, op: Op) -> Result<(), String> {
        let start_ns = time_ns();
        let d = dir_of(&op);
        if !self.cfg.real {
            let ssd = op.tier == "ssd" || op.host_tier.as_deref() == Some("ssd");
            let blocks = if ssd { op.slots.len() as i64 } else { op.b_hi - op.b_lo }; // an SSD op moves whole slots
            let nbytes = (blocks + op.state_slots.len() as i64) * self.cfg.slot_bytes;
            let bw = if ssd { self.cfg.ssd_read_gb_s } else { self.cfg.dram_read_gb_s } * 1e9;
            let due = {
                let mut g = lock(&self.due);
                let now = Instant::now();
                let base = g[d as usize].map_or(now, |t| t.max(now));
                let t = base + Duration::from_secs_f64(nbytes as f64 / bw);
                g[d as usize] = Some(t);
                t
            };
            let _ = self.q[d as usize].tx.send(QItem::Done(DoneItem {
                op_id: op.op_id, wait: Wait::Due(due), start_ns, nbytes, submit_ns: time_ns(), mark: None,
                recv_ns: op.recv_ns, trace: None, owned: vec![],
            }));
            return Ok(());
        }
        if op.tier == "ssd" {
            let _ = self.ssd_q[d as usize].tx.send(SsdItem::Op(op, start_ns));
            return Ok(());
        }
        if op.tier != "dram" || op.host_tier.is_some() || !(op.op == "read" || op.op == "write") {
            return Err(format!("ValueError: tier op {} {} via {:?}", op.op, op.tier, op.host_tier));
        }
        let mover = self.mover.as_ref().ok_or("no mover")?;
        let mut owned = Vec::new();
        let mut tr = (self.trace_on.load(Ordering::Relaxed) && op.op_id % self.cfg.trace_every.max(1) == 0)
            .then(Vec::new);
        let t0 = perf_ns();
        let base = self.cfg.dram_base.ok_or("no DRAM tier")?;
        let gate = || self.gate.yield_to_reads(&self.stop);
        let g: Option<&(dyn Fn() + Sync)> = if d == Dir::Read || op.urgent { None } else { Some(&gate) };
        let (ev, nbytes, mark) = self.move_(mover, d, base, &op, &op.slots, &op.state_slots, op.block_start,
                                            tr.as_mut(), &mut owned, g, true)?;
        let _ = self.q[d as usize].tx.send(QItem::Done(DoneItem {
            op_id: op.op_id, wait: ev.map_or(Wait::None, Wait::Ev), start_ns, nbytes, submit_ns: time_ns(), mark,
            recv_ns: op.recv_ns, trace: tr.map(|t| (op.op.clone(), t0, t)), owned,
        }));
        Ok(())
    }

    /// The op's blocks [b_lo, b_hi) and (if `sslots`) the state between HBM and the host slots at `base` (block i in
    /// slot slots[i - block_start]) (`PoolMember._move`).
    #[allow(clippy::too_many_arguments)]
    fn move_(&self, mover: &Mover, d: Dir, base: i64, op: &Op, slots: &[i64], sslots: &[i64], block_start: i64,
             tr: Option<&mut Vec<BatchTrace>>, owned: &mut Vec<Ev>, gate: Option<&(dyn Fn() + Sync)>, mark: bool)
        -> Result<(Option<Ev>, i64, Option<Ev>), String> {
        let p = self.placement.as_ref().ok_or("no placement")?;
        let (b_lo, b_hi) = (op.b_lo, op.b_hi);
        let sb = self.cfg.slot_bytes;
        let (dev, off, lens) = p.kv_pieces(&op.kv_blocks, b_lo, b_hi)?;
        let (a, b) = (b_lo - block_start, b_hi - block_start);
        if b > a && (a < 0 || b as usize > slots.len()) {
            return Err(format!("op {}: host slots [{a}, {b}) of {}", op.op_id, slots.len()));
        }
        let host: Vec<i64> = if b > a { slots[a as usize..b as usize].iter().map(|s| base + s * sb).collect() }
                             else { vec![] };
        let mut segs = vec![Segment { host_ptrs: host, dev, off, lens,
                                      local: Some(p.local(self.cfg.stage, b_hi - b_lo, false)) }];
        if !sslots.is_empty() {
            let (dev, off, lens) = p.state_pieces(&op.state_blocks, sslots.len())?;
            segs.push(Segment { host_ptrs: sslots.iter().map(|s| base + s * sb).collect(), dev, off, lens,
                                local: Some(p.local(self.cfg.stage, 0, true)) });
        }
        mover.transfer(d, &segs, tr, owned, gate, mark)
    }

    fn host_base(&self, op: &Op) -> Result<i64, String> {
        if op.host_tier.as_deref() == Some("dram") { self.cfg.dram_base.ok_or("no DRAM tier".into()) }
        else { self.cfg.staging_base.ok_or("no staging arena".into()) }
    }

    /// An SSD op, on its direction's SSD thread: file I/O on the io_uring, copies on the mover.
    fn ssd_op(self: &Arc<Self>, op: Op, start_ns: i64) -> Result<(), String> {
        let sb = self.cfg.slot_bytes;
        let base = self.host_base(&op)?;
        let files: Vec<i64> = op.slots.iter().chain(op.state_slots.iter()).copied().collect();
        let hosts: Vec<i64> = op.host_slots.iter().chain(op.host_state.iter()).map(|h| base + h * sb).collect();
        if files.len() != hosts.len() {
            return Err(format!("AssertionError: ({}, {}, {})", op.op_id, files.len(), hosts.len()));
        }
        let nbytes = files.len() as i64 * sb;
        match op.op.as_str() {
            "read" => self.file_io(op, start_ns, false, &files, &hosts, Landed::Copy),
            "fetch" => self.file_io(op, start_ns, false, &files, &hosts, Landed::Report(nbytes, Dir::Read)),
            "write" | "demote" => {
                if op.op == "write" {
                    // HBM -> host slots, then the file
                    let mover = self.mover.as_ref().ok_or("no mover")?;
                    let mut owned = Vec::new();
                    let gate = || self.gate.yield_to_reads(&self.stop);
                    let g: Option<&(dyn Fn() + Sync)> = if op.urgent { None } else { Some(&gate) };
                    let (ev, _, _) = self.move_(mover, Dir::Write, base, &op, &op.host_slots, &op.host_state,
                                                op.block_start, None, &mut owned, g, false)?;
                    if let Some(e) = ev {
                        cuda::synchronize(e)?;
                    }
                    mover.release(owned);
                }
                self.file_io(op, start_ns, true, &files, &hosts, Landed::Report(nbytes, Dir::Write))
            }
            other => Err(format!("ValueError: SSD op {other}")),
        }
    }

    /// Queue an op's file I/O (one request per slot, in order) and act once the last completed, on the ring's thread:
    /// a read goes to the copy thread (landing order, counted as a read to HBM then); writes, demotes and fetches
    /// are reported (the first failed I/O reports a WorkerError at once). The SSD thread does not wait for it.
    fn file_io(self: &Arc<Self>, op: Op, start_ns: i64, write: bool, files: &[i64], hosts: &[i64], then: Landed)
        -> Result<(), String> {
        let ssd = self.ssd.as_ref().ok_or("no SSD tier")?;
        let sb = self.cfg.slot_bytes;
        if files.is_empty() {
            match then {
                Landed::Copy => {
                    self.gate.queued(op.op_id);
                    let _ = self.copy_q.tx.send(CopyItem::Op(op, start_ns, 0, None));
                }
                Landed::Report(nbytes, d) => self.report_done(&op, start_ns, nbytes, d),
            }
            return Ok(());
        }
        *lock(&self.ssd_inflight) += 1;
        let t = Arc::new(Tracker { left: AtomicUsize::new(files.len() + 1), first_err: Mutex::new(None),
                                   failed: AtomicBool::new(false), abandoned: AtomicBool::new(false) });
        let op = Arc::new(op);
        let finish = {
            let (m, t, op) = (self.clone(), t.clone(), op.clone());
            move || {
                if t.abandoned.load(Ordering::SeqCst) {
                    return;
                }
                match then {
                    Landed::Copy => {
                        m.gate.queued(op.op_id);
                        let e = lock(&t.first_err).take().map(|x| x.1);
                        let _ = m.copy_q.tx.send(CopyItem::Op((*op).clone(), start_ns, files_len(&op), e));
                    }
                    Landed::Report(nbytes, d) => {
                        if !t.failed.load(Ordering::SeqCst) {
                            m.report_done(&op, start_ns, nbytes, d);
                        }
                    }
                }
                *lock(&m.ssd_inflight) -= 1;
                m.ssd_cv.notify_all();
            }
        };
        let finish = Arc::new(Mutex::new(Some(finish)));
        let one = {
            let (m, t, op, finish) = (self.clone(), t.clone(), op.clone(), finish.clone());
            move |i: usize, r: Result<(), String>| {
                if let Err(e) = r {
                    let first = !t.failed.swap(true, Ordering::SeqCst);
                    {
                        let mut fe = lock(&t.first_err);
                        if fe.as_ref().is_none_or(|x| i < x.0) {
                            *fe = Some((i, e.clone()));
                        }
                    }
                    if first && matches!(then, Landed::Report(..)) && !t.abandoned.load(Ordering::SeqCst) {
                        let d = if let Landed::Report(_, d) = then { d } else { Dir::Read };
                        let _ = m.q[d as usize].tx.send(QItem::Err(format!("tier SSD {} of op {}: {e}", op.op, op.op_id)));
                    }
                }
                if t.left.fetch_sub(1, Ordering::SeqCst) == 1 {
                    if let Some(f) = lock(&finish).take() {
                        f();
                    }
                }
            }
        };
        let one = Arc::new(one);
        for (i, (&f, &h)) in files.iter().zip(hosts).enumerate() {
            let o = one.clone();
            if let Err(e) = ssd.submit(write, f * sb, h, sb, Box::new(move |r| o(i, r))) {
                t.abandoned.store(true, Ordering::SeqCst);
                *lock(&self.ssd_inflight) -= 1;
                self.ssd_cv.notify_all();
                return Err(e);
            }
        }
        one(usize::MAX, Ok(())); // the guard: every request queued
        Ok(())
    }

    fn report_done(&self, op: &Op, start_ns: i64, nbytes: i64, d: Dir) {
        let _ = self.q[d as usize].tx.send(QItem::Done(DoneItem {
            op_id: op.op_id, wait: Wait::None, start_ns, nbytes, submit_ns: time_ns(), mark: None, recv_ns: op.recv_ns,
            trace: None, owned: vec![],
        }));
    }

    /// An SSD read whose slots all landed: HBM [b_lo, b_hi) and the state (host slots from block 0).
    fn ssd_read_copy(&self, op: Op, start_ns: i64, nslots: usize, err: Option<String>) -> Result<(), String> {
        if let Some(e) = err {
            return Err(e);
        }
        let base = self.host_base(&op)?;
        let mover = self.mover.as_ref().ok_or("no mover")?;
        let mut owned = Vec::new();
        let (ev, _, mark) = self.move_(mover, Dir::Read, base, &op, &op.host_slots, &op.host_state, 0, None,
                                       &mut owned, None, true)?;
        let _ = self.q[0].tx.send(QItem::Done(DoneItem {
            op_id: op.op_id, wait: ev.map_or(Wait::None, Wait::Ev), start_ns, nbytes: nslots as i64 * self.cfg.slot_bytes,
            submit_ns: time_ns(), mark, recv_ns: op.recv_ns, trace: None, owned,
        }));
        Ok(())
    }
}

fn files_len(op: &Op) -> usize {
    op.slots.len() + op.state_slots.len()
}

#[derive(Clone, Copy)]
enum Landed {
    Copy,
    Report(i64, Dir),
}

/// A done thread's GPU-to-host clock anchor: an event of its own recorded on an idle stream of its own and waited
/// for (so it ran, to the wake-up latency, just before the host read its clock), taken again every ANCHOR_S.
#[derive(Default)]
struct Anchor {
    stream: usize,
    ev: Option<Ev>,
    ns: i64,
    at: Option<Instant>,
}

impl Anchor {
    fn host_ns(&mut self, ev: Ev) -> Result<i64, String> {
        let now = Instant::now();
        if self.at.is_none_or(|a| (now - a).as_secs_f64() > ANCHOR_S) {
            if self.ev.is_none() {
                self.stream = cuda::stream_create()?;
                self.ev = Some(cuda::event_create(0)?);
            }
            let a = self.ev.unwrap();
            cuda::record(a, self.stream)?;
            cuda::synchronize(a)?;
            self.ns = time_ns();
            self.at = Some(now);
        }
        Ok(self.ns + (cuda::elapsed_ms(self.ev.unwrap(), ev)? as f64 * 1e6) as i64)
    }
}

fn anchor_drop() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_chunks_yield_to_reads_queued_then_not_later_ones() {
        let g = Arc::new(ReadGate { m: Mutex::new((0, 0, HashSet::new())), cv: Condvar::new(),
                                    waits: AtomicU64::new(0), wait_ms: Mutex::new(0.0) });
        let stop = Arc::new(AtomicBool::new(false));
        g.yield_to_reads(&stop); // no reads: no wait
        assert_eq!(g.waits.load(Ordering::Relaxed), 0);
        g.queued(1);
        g.queued(2);
        let passed = Arc::new(AtomicBool::new(false));
        let t = {
            let (g, stop, passed) = (g.clone(), stop.clone(), passed.clone());
            std::thread::spawn(move || {
                g.yield_to_reads(&stop);
                passed.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        g.queued(3); // later: does not hold the chunk back
        g.done(1);
        g.done(99); // not a counted read
        std::thread::sleep(Duration::from_millis(50));
        assert!(!passed.load(Ordering::SeqCst));
        g.done(2);
        t.join().unwrap();
        assert!(passed.load(Ordering::SeqCst));
        assert_eq!(g.waits.load(Ordering::Relaxed), 1);
    }
}
