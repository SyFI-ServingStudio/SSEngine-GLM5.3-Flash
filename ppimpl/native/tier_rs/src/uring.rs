//! `kvpool/uring.py` on the io-uring crate: reads and writes of byte ranges of one file into and out of registered
//! host memory, on one ring and one thread.
//!
//! - The file is registered (fixed file 0) and every host arena as fixed buffers of at most 1 GiB: READ_FIXED /
//!   WRITE_FIXED, the kernel pins nothing per request.
//! - A request is cut into pieces of at most `piece_bytes`, and at a fixed buffer's end.
//! - Reads first: pieces wait in two queues; at most `read_depth` read and `write_depth` write pieces are in the
//!   kernel at once, and free read room is filled before write room.
//! - One thread owns the ring (submit and wait in one call) and runs a request's callback when its last piece is in
//!   (on that thread, as the Python futures' callbacks). Others only queue and wake it through an eventfd whose read
//!   is always armed in the ring.

use std::collections::{HashMap, VecDeque};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use io_uring::{opcode, types, IoUring};

pub const MAX_BUFFER: i64 = 1 << 30; // kernel limit of one registered buffer
const WAKE: u64 = 0; // user_data of the eventfd read

pub type Cb = Box<dyn FnOnce(Result<(), String>) + Send>;

struct Req {
    left: usize,
    err: Option<String>,
    cb: Option<Cb>,
}

impl Req {
    fn fail(&mut self, e: &str) {
        if let Some(cb) = self.cb.take() {
            cb(Err(e.to_string()));
        }
    }
}

struct Piece {
    req: Arc<Mutex<Req>>,
    off: i64,
    addr: i64,
    n: i64,
    buf: usize,
}

struct Shared {
    q: Mutex<[VecDeque<Piece>; 2]>, // 0 reads, 1 writes
    efd: RawFd,
    stop: AtomicBool,
    error: Mutex<Option<String>>,
    stats: [AtomicU64; 2],
}

pub struct UringFile {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
    starts: Vec<i64>,
    ends: Vec<i64>,
    piece_bytes: i64,
    pub registered_bytes: i64,
    pub register_s: f64,
}

fn errstr(e: i32) -> String {
    std::io::Error::from_raw_os_error(e).to_string()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl UringFile {
    pub fn new(fd: RawFd, buffers: &[(i64, i64)], piece_bytes: i64, read_depth: usize, write_depth: usize)
        -> Result<UringFile, String> {
        if piece_bytes <= 0 || piece_bytes % 4096 != 0 || read_depth < 1 || write_depth < 1 {
            return Err(format!("io_uring piece {piece_bytes} B (a multiple of 4 KiB), depths {read_depth}/{write_depth}"));
        }
        let bits = usize::BITS - (read_depth + write_depth + 1).leading_zeros();
        let entries = 1u32 << bits;
        let ring = IoUring::new(entries).map_err(|e| format!("io_uring_queue_init: {e}"))?;
        ring.submitter().register_files(&[fd]).map_err(|e| format!("io_uring_register_files: {e}"))?;
        let mut spans: Vec<(i64, i64)> = buffers.iter()
            .flat_map(|&(a, n)| (0..n).step_by(MAX_BUFFER as usize).map(move |o| (a + o, MAX_BUFFER.min(n - o))))
            .collect();
        spans.sort();
        let iov: Vec<libc::iovec> = spans.iter()
            .map(|&(a, n)| libc::iovec { iov_base: a as *mut libc::c_void, iov_len: n as usize })
            .collect();
        let t0 = Instant::now();
        unsafe { ring.submitter().register_buffers(&iov) }.map_err(|e| format!(
            "io_uring_register_buffers ({} buffers, {:.1} GB): {e}", spans.len(),
            spans.iter().map(|s| s.1).sum::<i64>() as f64 / 1e9))?;
        let register_s = t0.elapsed().as_secs_f64();
        let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        if efd < 0 {
            return Err(format!("eventfd: {}", std::io::Error::last_os_error()));
        }
        let shared = Arc::new(Shared {
            q: Mutex::new([VecDeque::new(), VecDeque::new()]),
            efd,
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
            stats: [AtomicU64::new(0), AtomicU64::new(0)],
        });
        let sh = shared.clone();
        let depth = [read_depth, write_depth];
        let thread = std::thread::Builder::new().name("tier-uring".into())
            .spawn(move || run(ring, sh, depth)).map_err(|e| e.to_string())?;
        Ok(UringFile {
            shared,
            thread: Mutex::new(Some(thread)),
            starts: spans.iter().map(|s| s.0).collect(),
            ends: spans.iter().map(|s| s.0 + s.1).collect(),
            piece_bytes,
            registered_bytes: spans.iter().map(|s| s.1).sum(),
            register_s,
        })
    }

    /// `write` false: file -> addr; true: addr -> file. `cb` runs (on the ring's thread) once every piece completed.
    pub fn submit(&self, write: bool, offset: i64, addr: i64, nbytes: i64, cb: Cb) -> Result<(), String> {
        if let Some(e) = lock(&self.shared.error).as_ref() {
            return Err(format!("io_uring thread failed: {e}"));
        }
        let req = Arc::new(Mutex::new(Req { left: 0, err: None, cb: Some(cb) }));
        let mut pieces = Vec::new();
        let mut o = 0;
        while o < nbytes {
            let a = addr + o;
            let i = self.starts.partition_point(|&s| s <= a);
            if i == 0 || a >= self.ends[i - 1] {
                return Err(format!("io_uring: {a:#x} is in no registered buffer"));
            }
            let i = i - 1;
            let n = self.piece_bytes.min(nbytes - o).min(self.ends[i] - a);
            pieces.push(Piece { req: req.clone(), off: offset + o, addr: a, n, buf: i });
            o += n;
        }
        lock(&req).left = pieces.len();
        if pieces.is_empty() {
            let cb = lock(&req).cb.take().unwrap();
            cb(Ok(()));
            return Ok(());
        }
        lock(&self.shared.q)[write as usize].extend(pieces);
        wake(self.shared.efd);
        Ok(())
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.shared.stats[0].load(Ordering::Relaxed), self.shared.stats[1].load(Ordering::Relaxed))
    }

    pub fn close(&self) {
        let t = lock(&self.thread).take();
        if let Some(t) = t {
            self.shared.stop.store(true, Ordering::SeqCst);
            wake(self.shared.efd);
            let _ = t.join();
            unsafe {
                libc::close(self.shared.efd);
            }
        }
    }
}

impl Drop for UringFile {
    fn drop(&mut self) {
        self.close();
    }
}

fn wake(efd: RawFd) {
    let one: u64 = 1;
    unsafe {
        libc::write(efd, &one as *const u64 as *const libc::c_void, 8);
    }
}

fn run(mut ring: IoUring, sh: Arc<Shared>, depth: [usize; 2]) {
    let mut ebuf = Box::new([0u8; 8]);
    let mut live: HashMap<u64, (usize, Piece)> = HashMap::new();
    let mut inn = [0usize; 2];
    let mut next: u64 = 1;
    let res: Result<(), String> = (|| {
        let arm = |ring: &mut IoUring, ebuf: &mut [u8; 8]| -> Result<(), String> {
            let e = opcode::Read::new(types::Fd(sh.efd), ebuf.as_mut_ptr(), 8).build().user_data(WAKE);
            unsafe { ring.submission().push(&e) }.map_err(|_| "io_uring submission queue full".to_string())
        };
        arm(&mut ring, &mut ebuf)?;
        loop {
            {
                let mut q = lock(&sh.q);
                for k in 0..2 {
                    // reads take free room first
                    while inn[k] < depth[k] {
                        let Some(p) = q[k].pop_front() else { break };
                        let e = if k == 0 {
                            opcode::ReadFixed::new(types::Fixed(0), p.addr as *mut u8, p.n as u32, p.buf as u16)
                                .offset(p.off as u64).build()
                        } else {
                            opcode::WriteFixed::new(types::Fixed(0), p.addr as *const u8, p.n as u32, p.buf as u16)
                                .offset(p.off as u64).build()
                        };
                        let ud = next;
                        next += 1;
                        unsafe { ring.submission().push(&e.user_data(ud)) }
                            .map_err(|_| "io_uring submission queue full".to_string())?;
                        live.insert(ud, (k, p));
                        inn[k] += 1;
                    }
                }
            }
            if sh.stop.load(Ordering::SeqCst) && live.is_empty() {
                break;
            }
            match ring.submit_and_wait(1) {
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("io_uring_submit_and_wait: {e}")),
                Ok(_) => {}
            }
            let cqes: Vec<(u64, i32)> = ring.completion().map(|c| (c.user_data(), c.result())).collect();
            for (ud, r) in cqes {
                if ud == WAKE {
                    if r < 0 && r != -libc::EINTR {
                        return Err(format!("eventfd read: {}", errstr(-r)));
                    }
                    arm(&mut ring, &mut ebuf)?;
                    continue;
                }
                let Some((k, p)) = live.remove(&ud) else { continue };
                inn[k] -= 1;
                if r > 0 {
                    sh.stats[k].fetch_add(r as u64, Ordering::Relaxed);
                }
                let mut req = lock(&p.req);
                if r as i64 != p.n && req.err.is_none() {
                    let what = if k == 0 { "read" } else { "write" };
                    req.err = Some(if r < 0 {
                        format!("OSError({}, 'io_uring {what} at {}: {}')", -r, p.off, errstr(-r))
                    } else {
                        format!("OSError({}, 'io_uring {what} at {}: {r} of {} B')", libc::EIO, p.off, p.n)
                    });
                }
                req.left -= 1;
                if req.left == 0 {
                    let out = match req.err.take() {
                        None => Ok(()),
                        Some(e) => Err(e),
                    };
                    if let Some(cb) = req.cb.take() {
                        drop(req);
                        cb(out);
                    }
                }
            }
        }
        Ok(())
    })();
    if let Err(e) = res {
        // every waiting request fails with it
        *lock(&sh.error) = Some(e.clone());
        for (_, (_, p)) in live.drain() {
            lock(&p.req).fail(&e);
        }
        let mut q = lock(&sh.q);
        for k in 0..2 {
            for p in q[k].drain(..) {
                lock(&p.req).fail(&e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::AsRawFd;

    #[test]
    fn round_trip_through_two_buffers() {
        let dir = std::env::temp_dir().join(format!("tier_rs_uring_{}", std::process::id()));
        let mut f = std::fs::File::create(&dir).unwrap();
        f.write_all(&vec![0u8; 1 << 20]).unwrap();
        drop(f);
        let f = std::fs::OpenOptions::new().read(true).write(true).open(&dir).unwrap();
        let mut a = vec![0u8; 3 * 4096 + 4096];
        let mut b = vec![0u8; 3 * 4096 + 4096];
        let pa = a.as_mut_ptr() as i64 + (-(a.as_ptr() as i64)).rem_euclid(4096);
        let pb = b.as_mut_ptr() as i64 + (-(b.as_ptr() as i64)).rem_euclid(4096);
        for i in 0..3 * 4096 {
            unsafe { *((pa + i) as *mut u8) = (i % 253) as u8 };
        }
        let u = UringFile::new(f.as_raw_fd(), &[(pa, 3 * 4096), (pb, 3 * 4096)], 4096, 2, 1).unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let t = tx.clone();
        u.submit(true, 8192, pa, 3 * 4096, Box::new(move |r| t.send(r).unwrap())).unwrap();
        rx.recv().unwrap().unwrap();
        u.submit(false, 8192, pb, 3 * 4096, Box::new(move |r| tx.send(r).unwrap())).unwrap();
        rx.recv().unwrap().unwrap();
        for i in 0..3 * 4096 {
            assert_eq!(unsafe { *((pb + i) as *const u8) }, (i % 253) as u8);
        }
        assert!(u.submit(false, 0, pb + 3 * 4096, 4096, Box::new(|_| {})).is_err());
        drop(u);
        std::fs::remove_file(&dir).unwrap();
    }
}
