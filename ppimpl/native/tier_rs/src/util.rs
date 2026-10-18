//! Host clocks as Python reads them: `time.time_ns` (CLOCK_REALTIME), `time.perf_counter_ns` and
//! `time.monotonic` (CLOCK_MONOTONIC), `time.thread_time_ns` (CLOCK_THREAD_CPUTIME_ID).

fn clock(id: libc::clockid_t) -> i64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(id, &mut ts);
    }
    ts.tv_sec as i64 * 1_000_000_000 + ts.tv_nsec as i64
}

pub fn time_ns() -> i64 {
    clock(libc::CLOCK_REALTIME)
}

pub fn perf_ns() -> i64 {
    clock(libc::CLOCK_MONOTONIC)
}

pub fn thread_cpu_ns() -> i64 {
    clock(libc::CLOCK_THREAD_CPUTIME_ID)
}
