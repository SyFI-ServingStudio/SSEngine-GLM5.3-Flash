//! The driver calls of the tier threads, straight on libcuda (dlopen at first use, so the module imports on a host
//! without a driver): `cuMemcpyBatchAsync` at its CUDA 12.8 ABI (as vLLM's `swap_blocks_batch` and
//! `kvpool/mover.py`'s `Driver`: one attribute, source in stream order), event record / wait / sync / elapsed time,
//! and the primary context torch uses. No Python, no GIL.

use std::ffi::{c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;

pub type CuResult = c_int;
pub const CUDA_ERROR_NOT_READY: CuResult = 600;
pub const CU_EVENT_BLOCKING_SYNC: c_uint = 0x1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

/// A CUDA event handle (CUevent), copyable across threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ev(pub usize);

type Batch = unsafe extern "C" fn(
    *const u64, *const u64, *const usize, usize, *const u32, *const usize, usize, *mut usize, *mut c_void,
) -> CuResult;

pub struct Api {
    _lib: libloading::Library,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    primary_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult,
    primary_release: unsafe extern "C" fn(c_int) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(*mut c_void) -> CuResult,
    event_create: unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CuResult,
    event_destroy: unsafe extern "C" fn(*mut c_void) -> CuResult,
    event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> CuResult,
    event_sync: unsafe extern "C" fn(*mut c_void) -> CuResult,
    event_elapsed: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> CuResult,
    stream_wait: unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> CuResult,
    stream_create: unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CuResult,
    stream_destroy: unsafe extern "C" fn(*mut c_void) -> CuResult,
    stream_sync: unsafe extern "C" fn(*mut c_void) -> CuResult,
    batch: Batch,
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

pub fn api() -> Result<&'static Api, String> {
    API.get_or_init(load).as_ref().map_err(|e| e.clone())
}

fn load() -> Result<Api, String> {
    unsafe {
        let lib = libloading::Library::new("libcuda.so.1").map_err(|e| format!("libcuda.so.1: {e}"))?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get($name).map_err(|e| format!("{}: {e}", String::from_utf8_lossy($name)))?
            };
        }
        // cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus)
        let gpa: unsafe extern "C" fn(*const i8, *mut *mut c_void, c_int, u64, *mut c_int) -> CuResult =
            sym!(b"cuGetProcAddress_v2\0");
        let mut f: *mut c_void = std::ptr::null_mut();
        let mut status: c_int = 0;
        let name = CStr::from_bytes_with_nul(b"cuMemcpyBatchAsync\0").unwrap();
        let rc = gpa(name.as_ptr(), &mut f, 12080, 0, &mut status);
        if rc != 0 || f.is_null() {
            return Err(format!("cuMemcpyBatchAsync (CUDA 12.8 ABI) not available: CUresult {rc}"));
        }
        let batch: Batch = std::mem::transmute(f);
        Ok(Api {
            device_get: sym!(b"cuDeviceGet\0"),
            primary_retain: sym!(b"cuDevicePrimaryCtxRetain\0"),
            primary_release: sym!(b"cuDevicePrimaryCtxRelease_v2\0"),
            ctx_set_current: sym!(b"cuCtxSetCurrent\0"),
            event_create: sym!(b"cuEventCreate\0"),
            event_destroy: sym!(b"cuEventDestroy_v2\0"),
            event_record: sym!(b"cuEventRecord\0"),
            event_sync: sym!(b"cuEventSynchronize\0"),
            event_elapsed: sym!(b"cuEventElapsedTime\0"),
            stream_wait: sym!(b"cuStreamWaitEvent\0"),
            stream_create: sym!(b"cuStreamCreate\0"),
            stream_destroy: sym!(b"cuStreamDestroy_v2\0"),
            stream_sync: sym!(b"cuStreamSynchronize\0"),
            batch,
            _lib: lib,
        })
    }
}

fn ok(rc: CuResult, what: &str) -> Result<(), String> {
    if rc != 0 {
        Err(format!("{what}: CUresult {rc}"))
    } else {
        Ok(())
    }
}

/// The device's primary context (the one torch made current in the worker), retained for this process.
pub struct Ctx {
    pub ctx: usize,
    pub ordinal: i32,
}

impl Ctx {
    pub fn retain(ordinal: i32) -> Result<Ctx, String> {
        let a = api()?;
        let mut dev: c_int = 0;
        let mut ctx: *mut c_void = std::ptr::null_mut();
        unsafe {
            ok((a.device_get)(&mut dev, ordinal), "cuDeviceGet")?;
            ok((a.primary_retain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
        }
        Ok(Ctx { ctx: ctx as usize, ordinal: dev })
    }

    /// Make the context current on the calling thread (each tier thread, once).
    pub fn make_current(&self) -> Result<(), String> {
        unsafe { ok((api()?.ctx_set_current)(self.ctx as *mut c_void), "cuCtxSetCurrent") }
    }

    pub fn release(&self) {
        if let Ok(a) = api() {
            unsafe {
                (a.primary_release)(self.ordinal);
            }
        }
    }
}

pub fn event_create(flags: c_uint) -> Result<Ev, String> {
    let mut e: *mut c_void = std::ptr::null_mut();
    unsafe { ok((api()?.event_create)(&mut e, flags), "cuEventCreate")? };
    Ok(Ev(e as usize))
}

pub fn event_destroy(e: Ev) {
    if let Ok(a) = api() {
        unsafe {
            (a.event_destroy)(e.0 as *mut c_void);
        }
    }
}

pub fn record(e: Ev, stream: usize) -> Result<(), String> {
    unsafe { ok((api()?.event_record)(e.0 as *mut c_void, stream as *mut c_void), "cuEventRecord") }
}

pub fn wait(stream: usize, e: Ev) -> Result<(), String> {
    unsafe { ok((api()?.stream_wait)(stream as *mut c_void, e.0 as *mut c_void, 0), "cuStreamWaitEvent") }
}

pub fn synchronize(e: Ev) -> Result<(), String> {
    unsafe { ok((api()?.event_sync)(e.0 as *mut c_void), "cuEventSynchronize") }
}

/// ms from event `a` to event `b`, both completed; NOT_READY (not expected: callers read completed events) waits
/// for both, then reads again (as `Driver.elapsed_ms`).
pub fn elapsed_ms(a: Ev, b: Ev) -> Result<f32, String> {
    let api = api()?;
    let mut ms: f32 = 0.0;
    unsafe {
        let mut rc = (api.event_elapsed)(&mut ms, a.0 as *mut c_void, b.0 as *mut c_void);
        if rc == CUDA_ERROR_NOT_READY {
            synchronize(a)?;
            synchronize(b)?;
            rc = (api.event_elapsed)(&mut ms, a.0 as *mut c_void, b.0 as *mut c_void);
        }
        ok(rc, "cuEventElapsedTime")?;
    }
    Ok(ms)
}

pub fn stream_create() -> Result<usize, String> {
    let mut s: *mut c_void = std::ptr::null_mut();
    unsafe { ok((api()?.stream_create)(&mut s, CU_STREAM_NON_BLOCKING), "cuStreamCreate")? };
    Ok(s as usize)
}

pub fn stream_destroy(s: usize) {
    if let Ok(a) = api() {
        unsafe {
            (a.stream_destroy)(s as *mut c_void);
        }
    }
}

pub fn stream_sync(s: usize) -> Result<(), String> {
    unsafe { ok((api()?.stream_sync)(s as *mut c_void), "cuStreamSynchronize") }
}

// CUmemcpyAttributes (CUDA 12.8): srcAccessOrder = STREAM (1), no location hints, no flags
static ATTR: [u32; 6] = [1, 0, 0, 0, 0, 0];
static ATTR_IDX: [usize; 1] = [0];

/// One `cuMemcpyBatchAsync` of src[i] -> dst[i], lens[i] bytes on `stream`.
pub fn batch(stream: usize, src: &[i64], dst: &[i64], lens: &[i64]) -> Result<(), String> {
    let a = api()?;
    let mut fail: usize = 0;
    let rc = unsafe {
        (a.batch)(
            dst.as_ptr() as *const u64,
            src.as_ptr() as *const u64,
            lens.as_ptr() as *const usize,
            lens.len(),
            ATTR.as_ptr(),
            ATTR_IDX.as_ptr(),
            1,
            &mut fail,
            stream as *mut c_void,
        )
    };
    if rc != 0 {
        return Err(format!("cuMemcpyBatchAsync of {} copies failed at {fail}: CUresult {rc}", lens.len()));
    }
    Ok(())
}
