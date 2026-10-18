//! `tier_rs`: the worker's KV-tier execution (lanes/m6-rust TIER_RS.md). Python builds the arenas, the placement,
//! the ring and the streams at start-up and hands their raw values to `Engine`; from `start()` to `close()` every
//! tier op runs on Rust threads that never take the GIL. Every call from Python releases the GIL while it works.

pub mod cuda;
pub mod member;
pub mod mover;
pub mod placement;
pub mod uring;
pub mod util;
pub mod wire;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use member::{lock, Config, Member, Report};
use mover::{Backend, Dir, Mover, Segment};
use placement::Placement;
use uring::UringFile;

fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
    d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("tier_rs.Engine: missing {k}")))?.extract()
}

fn opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Option<T>> {
    match d.get_item(k)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract()?)),
    }
}

#[pyclass]
struct Engine {
    m: Arc<Member>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    closed: Mutex<bool>,
}

#[pymethods]
impl Engine {
    /// `cfg` (all start-up values, TIER_RS.md): stage, real, cuda, device, slot_bytes, dram_base, staging_base,
    /// trace_every, run_dir, dram_read_gb_s, ssd_read_gb_s; placement arrays (kv_stage,
    /// kv_dev, kv_stride, kv_len, kv_host, st_stage, st_dev, st_stride, st_group, st_slot, st_off, st_len);
    /// hops, ring_ptr, ring_depth, ring_chunk_slots, streams [read, write, read_nvl, write_nvl];
    /// ssd_fd, ssd_buffers [(addr, bytes)], ssd_piece_bytes, ssd_read_depth, ssd_write_depth.
    #[new]
    fn new(py: Python<'_>, cfg: &Bound<'_, PyDict>) -> PyResult<Self> {
        let real: bool = get(cfg, "real")?;
        let cuda_on: bool = get(cfg, "cuda")?;
        let slot_bytes: i64 = get(cfg, "slot_bytes")?;
        let c = Config {
            stage: get(cfg, "stage")?,
            real,
            slot_bytes,
            dram_base: opt(cfg, "dram_base")?,
            staging_base: opt(cfg, "staging_base")?,
            trace_every: get(cfg, "trace_every")?,
            run_dir: get(cfg, "run_dir")?,
            dram_read_gb_s: get(cfg, "dram_read_gb_s")?,
            ssd_read_gb_s: get(cfg, "ssd_read_gb_s")?,
        };
        let placement = if real {
            Some(Placement {
                slot_bytes,
                kv_stage: get(cfg, "kv_stage")?,
                kv_dev: get(cfg, "kv_dev")?,
                kv_stride: get(cfg, "kv_stride")?,
                kv_len: get(cfg, "kv_len")?,
                kv_host: get(cfg, "kv_host")?,
                st_stage: get(cfg, "st_stage")?,
                st_dev: get(cfg, "st_dev")?,
                st_stride: get(cfg, "st_stride")?,
                st_group: get(cfg, "st_group")?,
                st_slot: get(cfg, "st_slot")?,
                st_off: get(cfg, "st_off")?,
                st_len: get(cfg, "st_len")?,
            })
        } else {
            None
        };
        let device: i32 = get(cfg, "device")?;
        let two = match get::<String>(cfg, "hops")?.as_str() {
            "two" => true,
            "direct" => false,
            h => return Err(PyValueError::new_err(format!("tiers.hops: {h:?} (two | direct)"))),
        };
        let depth: usize = get(cfg, "ring_depth")?;
        let chunk: usize = get(cfg, "ring_chunk_slots")?;
        let ring_ptr: Option<i64> = opt(cfg, "ring_ptr")?;
        let streams: Option<Vec<usize>> = opt(cfg, "streams")?;
        let ssd: Option<(i32, Vec<(i64, i64)>, i64, usize, usize)> = match opt::<i32>(cfg, "ssd_fd")? {
            None => None,
            Some(fd) => Some((fd, get(cfg, "ssd_buffers")?, get(cfg, "ssd_piece_bytes")?, get(cfg, "ssd_read_depth")?,
                              get(cfg, "ssd_write_depth")?)),
        };
        let m = py.allow_threads(move || -> Result<Member, String> {
            let ctx = if cuda_on { Some(cuda::Ctx::retain(device)?) } else { None };
            if let Some(c) = &ctx {
                c.make_current()?;
            }
            let mover = if real {
                let backend = match (&ctx, streams) {
                    (Some(_), Some(s)) if s.len() == 4 => Backend::Cuda { streams: [s[0], s[1], s[2], s[3]] },
                    (Some(_), _) => return Err("CUDA mover: four streams [read, write, read_nvl, write_nvl]".into()),
                    (None, _) => Backend::Cpu,
                };
                Some(Mover::new(backend, slot_bytes, two, depth, chunk, ring_ptr)?)
            } else {
                None
            };
            let ssd = match ssd {
                None => None,
                Some((fd, bufs, piece, rd, wd)) => Some(UringFile::new(fd, &bufs, piece, rd, wd)?),
            };
            Ok(Member::new(c, placement, mover, ssd, ctx))
        }).map_err(err)?;
        Ok(Engine { m: Arc::new(m), threads: Mutex::new(Vec::new()), closed: Mutex::new(false) })
    }

    /// Start the threads. `tier_addr`: the receive thread binds it (None in tests that `dispatch`); `report_addr`: the
    /// done threads push to it (None: completions for `poll_done`).
    #[pyo3(signature = (tier_addr = None, report_addr = None))]
    fn start(&self, py: Python<'_>, tier_addr: Option<String>, report_addr: Option<String>) -> PyResult<()> {
        let recv = tier_addr.is_some();
        *lock(&self.m.addrs) = (tier_addr, report_addr);
        py.allow_threads(|| {
            let mut ts = lock(&self.threads);
            let spawn = |ts: &mut Vec<JoinHandle<()>>, name: &str, f: Box<dyn FnOnce() + Send>| -> Result<(), String> {
                ts.push(std::thread::Builder::new().name(name.to_string()).spawn(f).map_err(|e| e.to_string())?);
                Ok(())
            };
            for (d, n) in [(Dir::Read, "read"), (Dir::Write, "write")] {
                let m = self.m.clone();
                spawn(&mut ts, &format!("tier-{n}"), Box::new(move || m.done_loop(d)))?;
                let m = self.m.clone();
                spawn(&mut ts, &format!("tier-submit-{n}"), Box::new(move || m.submit_loop(d)))?;
                if self.m.ssd.is_some() {
                    let m = self.m.clone();
                    spawn(&mut ts, &format!("tier-ssd-{n}"), Box::new(move || m.ssd_loop(d)))?;
                }
            }
            if self.m.ssd.is_some() {
                let m = self.m.clone();
                spawn(&mut ts, "tier-ssd-copy", Box::new(move || m.copy_loop()))?;
            }
            if recv {
                let m = self.m.clone();
                spawn(&mut ts, "tier-recv", Box::new(move || m.recv_loop()))?;
            }
            let m = self.m.clone();
            spawn(&mut ts, "tier-trace", Box::new(move || m.trace_loop()))?;
            Ok(())
        }).map_err(err)
    }

    /// Tests: one tier message as the receive thread handles it (a Shutdown ends the receive side).
    fn dispatch(&self, py: Python<'_>, buf: &[u8]) -> PyResult<()> {
        let buf = buf.to_vec();
        py.allow_threads(|| match self.m.handle(&buf) {
            Ok(true) => Ok(()),
            Ok(false) => {
                self.m.end_chain(None);
                Ok(())
            }
            Err(e) => Err(e),
        }).map_err(err)
    }

    /// Tests (report=None): the next completion as a dict (TierDone fields, or {"type": "WorkerError", "error"}), or
    /// None after `timeout` s.
    fn poll_done(&self, py: Python<'_>, timeout: f64) -> PyResult<Option<PyObject>> {
        let r = py.allow_threads(|| self.m.local.rx.recv_timeout(Duration::from_secs_f64(timeout)).ok());
        let Some(r) = r else { return Ok(None) };
        let d = PyDict::new(py);
        match r {
            Report::Done(x) => {
                d.set_item("type", "TierDone")?;
                d.set_item("stage", x.stage)?;
                d.set_item("op_id", x.op_id)?;
                d.set_item("start_ns", x.start_ns)?;
                d.set_item("end_ns", x.end_ns)?;
                d.set_item("nbytes", x.nbytes)?;
                d.set_item("submit_ns", x.submit_ns)?;
                d.set_item("gpu_ns", x.gpu_ns)?;
                d.set_item("mark_ns", x.mark_ns)?;
                d.set_item("recv_ns", x.recv_ns)?;
            }
            Report::Error(e) => {
                d.set_item("type", "WorkerError")?;
                d.set_item("stage", self.m.cfg.stage)?;
                d.set_item("error", e)?;
            }
        }
        Ok(Some(d.into_any().unbind()))
    }

    /// Stop: the receive side ends (its thread sees the stop within 200 ms), every thread drains and is joined (up to
    /// 10 s each), the io_uring stops. Returns the counters the Python member prints at close.
    fn close(&self, py: Python<'_>) -> PyResult<PyObject> {
        let joined = py.allow_threads(|| {
            let mut closed = lock(&self.closed);
            if *closed {
                return true;
            }
            *closed = true;
            self.m.stop.store(true, Ordering::SeqCst);
            let ts: Vec<JoinHandle<()>> = lock(&self.threads).drain(..).collect();
            let has_recv = ts.iter().any(|t| t.thread().name() == Some("tier-recv"));
            if !has_recv {
                self.m.end_chain(None);
            }
            let mut all = true;
            for t in ts {
                let deadline = Instant::now() + Duration::from_secs(10);
                while !t.is_finished() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
                if t.is_finished() {
                    let _ = t.join();
                } else {
                    all = false;
                }
            }
            all
        });
        let m = &self.m;
        let d = PyDict::new(py);
        d.set_item("joined", joined)?;
        let (waits, wait_ms) = m.yields();
        d.set_item("yield_waits", waits)?;
        d.set_item("yield_wait_ms", wait_ms)?;
        if let Some(mv) = &m.mover {
            d.set_item("events_made", mv.events_made.load(Ordering::Relaxed))?;
            d.set_item("record_ns", lock(&mv.record_ns).clone())?;
            d.set_item("wait_ns", lock(&mv.wait_ns).clone())?;
            d.set_item("gate_wait_ns", lock(&mv.gate_wait_ns).clone())?;
        }
        if let Some(u) = &m.ssd {
            let (r, w) = u.stats();
            d.set_item("ssd_read_bytes", r)?;
            d.set_item("ssd_write_bytes", w)?;
        }
        py.allow_threads(|| {
            if let Some(u) = &m.ssd {
                u.close();
            }
            if joined {
                if let Some(mv) = &m.mover {
                    mv.destroy_events();
                }
            }
        });
        Ok(d.into_any().unbind())
    }

    #[getter]
    fn ssd_info(&self) -> Option<(i64, f64)> {
        self.m.ssd.as_ref().map(|u| (u.registered_bytes, u.register_s))
    }
}

/// Tests: the batches the mover issues for one transfer, [(stream, src, dst, lens)], without copying (`ring_ptr`:
/// the Python mover's ring, so both carry the same addresses; `ring_pos`: the direction's next ring chunk). `segments`: [(host_ptrs, dev, off, lens, local)].
#[pyfunction]
#[pyo3(signature = (direction, segments, slot_bytes, hops, ring_depth, ring_chunk_slots, ring_ptr, gated, ring_pos = 0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn plan_transfer(py: Python<'_>, direction: &str, segments: Vec<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Option<Vec<bool>>)>,
                 slot_bytes: i64, hops: &str, ring_depth: usize, ring_chunk_slots: usize, ring_ptr: Option<i64>,
                 gated: bool, ring_pos: usize) -> PyResult<Vec<(String, Vec<i64>, Vec<i64>, Vec<i64>)>> {
    let d = if direction == "read" { Dir::Read } else { Dir::Write };
    let segs: Vec<Segment> = segments.into_iter()
        .map(|(h, dv, o, l, loc)| Segment { host_ptrs: h, dev: dv, off: o, lens: l, local: loc }).collect();
    let recs = py.allow_threads(|| -> Result<_, String> {
        let m = Mover::new(Backend::Record(Mutex::new(vec![])), slot_bytes, hops == "two", ring_depth, ring_chunk_slots,
                           ring_ptr)?;
        m.set_ring_next(d, ring_pos);
        let gate = || {};
        let g: Option<&(dyn Fn() + Sync)> = if gated { Some(&gate) } else { None };
        m.transfer(d, &segs, None, &mut vec![], g, true)?;
        Ok(m.take_records())
    }).map_err(err)?;
    Ok(recs.into_iter().map(|r| (mover::STREAM_NAMES[r.stream].to_string(), r.src, r.dst, r.lens)).collect())
}

/// Tests: decode a tier message and encode its TierDone the way the threads do (wire round trip).
#[pyfunction]
fn wire_echo(py: Python<'_>, buf: &[u8]) -> PyResult<PyObject> {
    match wire::decode(buf).map_err(err)? {
        wire::Msg::Shutdown => Ok(py.None()),
        wire::Msg::Batch(ops) => {
            let out: Vec<Py<PyBytes>> = ops.iter().map(|o| {
                let d = wire::Done { stage: o.b_lo, op_id: o.op_id, start_ns: o.b_hi, end_ns: o.block_start,
                                     nbytes: o.slots.iter().sum(), submit_ns: o.urgent as i64,
                                     gpu_ns: o.host_tier.as_ref().map(|s| s.len() as i64), mark_ns: None,
                                     recv_ns: Some(o.kv_blocks.iter().map(|b| b.len() as i64).sum()) };
                PyBytes::new(py, &wire::encode_done(&d)).unbind()
            }).collect();
            Ok(out.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pymodule]
fn tier_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(plan_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(wire_echo, m)?)?;
    Ok(())
}
