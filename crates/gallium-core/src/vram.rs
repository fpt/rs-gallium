//! One owner for an accelerator's free memory (issue #343).
//!
//! Every device consumer used to size itself without knowing about the others:
//! the [`ExpertCache`] from a fixed `expertCacheBytes`, the KV cache by doubling
//! whenever it filled, the prefill transient not at all. They met at the
//! driver, as `CUDA_ERROR_OUT_OF_MEMORY` — and on candle's CUDA stack an OOM
//! is not recoverable in-process (the VRAM stays unreleased and the context is
//! broken; see `responses_api.rs`'s module doc). So the only fix is that the
//! driver never sees an allocation that will fail, which needs one ledger that
//! every allocation is booked against **before** it is made.
//!
//! Priorities, highest first:
//!
//! 1. resident weights — already allocated when the ledger is made, so the
//!    budget is what is free *after* them;
//! 2. KV cache, and 3. the prefill transient — [`VramLedger::reserve`] before
//!    allocating;
//! 4. the [`ExpertCache`] — elastic: it holds whatever is not reserved, and is
//!    evicted to make room before a reservation is refused. Experts are
//!    mmap-backed, so evicting one only drops the device copy.
//!
//! The ledger counts bytes; it does not ask the driver. cudarc allocates
//! through `cuMemAllocAsync` whenever the device supports memory pools (every
//! card this runs on), so a freed buffer goes back to the stream's pool, not to
//! the driver, and `cuMemGetInfo` under-reports what is actually available
//! between synchronizations. The driver is read exactly once, at load, after a
//! synchronize ([`free_device_memory`]).
//!
//! A reservation that cannot fit even with the cache emptied is refused with
//! [`VramExhausted`] — an ordinary error, raised before any device memory is
//! touched, so the process and its CUDA context survive it.

use std::sync::{Arc, Mutex};

use candle_core::{Device, Result};

use crate::ExpertCache;

/// The device-memory budget for one loaded model, and what has been reserved
/// against it. Shared (`Arc`) by everything that allocates on the device.
pub struct VramLedger {
    budget: usize,
    state: Mutex<LedgerState>,
    expert_cache: Option<Arc<ExpertCache>>,
}

struct LedgerState {
    /// Held until released: the KV cache and its checkpoints.
    persistent: usize,
    /// Held for one forward: the prefill/decode scratch.
    transient: usize,
    /// The most `transient` has ever been. Kept away from the expert cache
    /// for good, rather than evicted for and given back on every forward —
    /// the transient outranks the cache (issue #343's priorities), and paying
    /// for it per decode token would evict a few experts every token only to
    /// upload them again on the next.
    headroom: usize,
    /// Highest `persistent + transient` ever reached — what a run needed.
    high_water: usize,
}

impl LedgerState {
    fn live(&self) -> usize {
        self.persistent + self.transient
    }
}

/// A refused [`VramLedger::reserve`]: `wanted` bytes for `what`, with only
/// `available` left after evicting the whole expert cache.
#[derive(Debug, Clone)]
pub struct VramExhausted {
    pub what: &'static str,
    pub wanted: usize,
    pub available: usize,
    pub budget: usize,
}

impl std::fmt::Display for VramExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{VRAM_EXHAUSTED}: {} needs {} MiB, {} MiB of the {} MiB budget left \
             (expert cache already emptied) — the context does not fit this device; \
             compact the conversation or lower maxCtx",
            self.what,
            self.wanted >> 20,
            self.available >> 20,
            self.budget >> 20,
        )
    }
}

impl std::error::Error for VramExhausted {}

/// The phrase a [`VramExhausted`] always starts with, for callers that only see
/// the error as text by the time it reaches them.
pub const VRAM_EXHAUSTED: &str = "VRAM budget exhausted";

/// Whether `err` is (or wraps) a [`VramExhausted`].
pub fn is_vram_exhausted(err: &candle_core::Error) -> bool {
    err.to_string().contains(VRAM_EXHAUSTED)
}

impl VramLedger {
    /// A ledger over `budget` bytes. An attached `expert_cache` is handed the
    /// whole budget at once and gives it back as reservations are made.
    pub fn new(budget: usize, expert_cache: Option<Arc<ExpertCache>>) -> Arc<Self> {
        if let Some(cache) = &expert_cache {
            cache.set_budget(budget);
        }
        Arc::new(Self {
            budget,
            state: Mutex::new(LedgerState {
                persistent: 0,
                transient: 0,
                headroom: 0,
                high_water: 0,
            }),
            expert_cache,
        })
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Everything booked right now, persistent and transient.
    pub fn reserved(&self) -> usize {
        self.state.lock().unwrap().live()
    }

    /// What is held back from the expert cache for transients — see
    /// `LedgerState::headroom`.
    pub fn headroom(&self) -> usize {
        self.state.lock().unwrap().headroom
    }

    pub fn high_water(&self) -> usize {
        self.state.lock().unwrap().high_water
    }

    pub fn expert_cache(&self) -> Option<&Arc<ExpertCache>> {
        self.expert_cache.as_ref()
    }

    /// Book `bytes` for `what` — held until the [`Reservation`] drops — before
    /// allocating them. The expert cache is shrunk to whatever remains first,
    /// so on success the memory is free by the time the caller allocates.
    /// Refused, with nothing changed, when it does not fit even with the cache
    /// empty.
    pub fn reserve(self: &Arc<Self>, bytes: usize, what: &'static str) -> Result<Reservation> {
        self.book(bytes, what, false)
    }

    /// [`Self::reserve`] for scratch that lives through one forward. It moves
    /// the expert cache only when it is the largest yet — see
    /// `LedgerState::headroom`.
    pub fn reserve_transient(
        self: &Arc<Self>,
        bytes: usize,
        what: &'static str,
    ) -> Result<Reservation> {
        self.book(bytes, what, true)
    }

    fn book(
        self: &Arc<Self>,
        bytes: usize,
        what: &'static str,
        transient: bool,
    ) -> Result<Reservation> {
        let mut st = self.state.lock().unwrap();
        self.admit(&st, bytes, what)?;
        if transient {
            st.transient += bytes;
            st.headroom = st.headroom.max(st.transient);
        } else {
            st.persistent += bytes;
        }
        st.high_water = st.high_water.max(st.live());
        self.sync_cache(&st);
        Ok(Reservation {
            ledger: self.clone(),
            bytes,
            transient,
        })
    }

    /// Refuse `bytes` more if the live total would pass the budget. Nothing is
    /// changed on refusal — the cache keeps what it has: a refused request
    /// should not also cost the next decode its experts.
    fn admit(&self, st: &LedgerState, bytes: usize, what: &'static str) -> Result<()> {
        let available = self.budget.saturating_sub(st.live());
        if bytes > available {
            return Err(candle_core::Error::wrap(VramExhausted {
                what,
                wanted: bytes,
                available,
                budget: self.budget,
            }));
        }
        Ok(())
    }

    /// Change a reservation of `*held` bytes to `bytes`. Growing can be
    /// refused, exactly as [`Self::reserve`] can; shrinking cannot.
    fn resize(
        &self,
        held: &mut usize,
        transient: bool,
        bytes: usize,
        what: &'static str,
    ) -> Result<()> {
        let mut st = self.state.lock().unwrap();
        if bytes > *held {
            self.admit(&st, bytes - *held, what)?;
        }
        let slot = if transient {
            &mut st.transient
        } else {
            &mut st.persistent
        };
        *slot = *slot - *held + bytes;
        if transient {
            st.headroom = st.headroom.max(st.transient);
        }
        st.high_water = st.high_water.max(st.live());
        *held = bytes;
        self.sync_cache(&st);
        Ok(())
    }

    fn release(&self, bytes: usize, transient: bool) {
        let mut st = self.state.lock().unwrap();
        let slot = if transient {
            &mut st.transient
        } else {
            &mut st.persistent
        };
        *slot = slot.saturating_sub(bytes);
        self.sync_cache(&st);
    }

    /// Called with the state lock held, so a cache budget can never be set
    /// from a stale state. Lock order: ledger, then cache — the cache never
    /// calls back into the ledger.
    fn sync_cache(&self, st: &LedgerState) {
        if let Some(cache) = &self.expert_cache {
            let held_back = st.persistent + st.headroom.max(st.transient);
            cache.set_budget(self.budget.saturating_sub(held_back));
        }
    }
}

/// Bytes booked against a [`VramLedger`], returned to it on drop — and from
/// there to the expert cache.
pub struct Reservation {
    ledger: Arc<VramLedger>,
    bytes: usize,
    transient: bool,
}

impl Reservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Hold `bytes` instead — growing can be refused, shrinking cannot.
    pub fn resize(&mut self, bytes: usize, what: &'static str) -> Result<()> {
        self.ledger
            .resize(&mut self.bytes, self.transient, bytes, what)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.release(self.bytes, self.transient);
    }
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reservation")
            .field("bytes", &self.bytes)
            .field("transient", &self.transient)
            .finish()
    }
}

/// Book `bytes` on `ledger` if there is one; `None` otherwise. The shape every
/// allocation site uses, so "no ledger" stays today's unaccounted behaviour.
pub fn reserve_on(
    ledger: Option<&Arc<VramLedger>>,
    bytes: usize,
    what: &'static str,
) -> Result<Option<Reservation>> {
    ledger.map(|l| l.reserve(bytes, what)).transpose()
}

/// The pool granularity measured on an RTX 4070 (`device_charge` below): every
/// `cuMemAllocAsync` buffer is charged in multiples of 512 KiB, with 512 KiB
/// the minimum. An expert's ~1.1 MB down projection costs 1.5 MiB and its
/// ~2.2 MB gate/up 2.5 MiB — 1.24x the bytes in aggregate, which is exactly
/// the margin an expert cache counting raw bytes overran the card by.
const CUDA_ALLOC_GRANULARITY: usize = 512 << 10;

/// What allocating a `bytes`-long buffer on `device` actually costs: rounded
/// up to the CUDA pool's granularity, as-is anywhere else. The ledger books
/// this, never the tensor's own size.
pub fn device_charge(bytes: usize, device: &Device) -> usize {
    if device.is_cuda() && bytes > 0 {
        bytes.div_ceil(CUDA_ALLOC_GRANULARITY) * CUDA_ALLOC_GRANULARITY
    } else {
        bytes
    }
}

/// Free memory on `device` as the driver reports it, after a synchronize so
/// pending frees have landed. `None` for a device with no such query (CPU,
/// Metal — whose memory is the host's, which is not a budget to divide).
pub fn free_device_memory(device: &Device) -> Result<Option<usize>> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda) => {
            device.synchronize()?;
            let (free, _total) = cuda
                .cuda_stream()
                .context()
                .mem_get_info()
                .map_err(|e| candle_core::Error::Msg(format!("cuMemGetInfo: {e}")))?;
            Ok(Some(free))
        }
        _ => {
            let _ = device;
            Ok(None)
        }
    }
}

/// A CUDA memory pool of its own, for the [`ExpertCache`] (issue #343).
///
/// Every candle allocation goes through `cuMemAllocAsync` on the device's
/// current pool. With one pool for everything, the cache's thousands of ~2 MiB,
/// long-lived, constantly churning entries end up scattered through the chunks
/// a prefill's 8–16 MB f32 expert expansions forced the pool to map — and a
/// chunk with one live entry in it cannot go back to the driver. Measured on
/// the 4070: the pool's *used* bytes held at ~7.9 GB (what the ledger counts)
/// while its *reserved* bytes climbed by 300–500 MB per prefill chunk to 11.5
/// GB, `cuMemPoolTrimTo(0)` recovering none of it, until the driver refused.
/// No byte count can see that, so the cache's buffers live in their own pool:
/// its entries come in a couple of sizes and reuse each other's blocks, the
/// default pool keeps the short-lived transients and trims cleanly, and a
/// shrinking cache can [`trim`](Self::trim) to hand memory back to the driver.
#[cfg(feature = "cuda")]
pub struct DevicePool {
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    ctx: Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
    device: Device,
}

// The handle is an opaque driver object, usable from any thread bound to the
// context; every use below binds first.
#[cfg(feature = "cuda")]
unsafe impl Send for DevicePool {}
#[cfg(feature = "cuda")]
unsafe impl Sync for DevicePool {}

#[cfg(feature = "cuda")]
impl DevicePool {
    /// A fresh pool on `device`; `None` on anything but CUDA.
    pub fn new(device: &Device) -> Result<Option<Self>> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let Device::Cuda(cuda) = device else {
            return Ok(None);
        };
        let ctx = cuda.cuda_stream().context().clone();
        ctx.bind_to_thread().map_err(cuda_err("bind context"))?;
        unsafe {
            let mut props: sys::CUmemPoolProps = std::mem::zeroed();
            props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
            props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
            // `location.id` is a plain field before CUDA 13.2 and the first
            // member of a union after; either way it is the `c_int` right after
            // the 4-byte `type_`.
            std::ptr::addr_of_mut!(props.location)
                .cast::<u8>()
                .add(4)
                .cast::<i32>()
                .write(ctx.ordinal() as i32);
            let mut pool = std::ptr::null_mut();
            sys::cuMemPoolCreate(&mut pool, &props)
                .result()
                .map_err(cuda_err("cuMemPoolCreate"))?;
            Ok(Some(Self {
                pool,
                ctx,
                device: device.clone(),
            }))
        }
    }

    /// Run `f` with this pool as the device's current one, so every buffer it
    /// allocates comes from here. Frees go back to whichever pool a buffer
    /// came from, so only the allocation needs the scope.
    pub fn scope<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        self.ctx
            .bind_to_thread()
            .map_err(cuda_err("bind context"))?;
        let dev = self.ctx.cu_device();
        let mut previous = std::ptr::null_mut();
        unsafe {
            sys::cuDeviceGetMemPool(&mut previous, dev)
                .result()
                .map_err(cuda_err("cuDeviceGetMemPool"))?;
            sys::cuDeviceSetMemPool(dev, self.pool)
                .result()
                .map_err(cuda_err("cuDeviceSetMemPool"))?;
        }
        let out = f();
        unsafe {
            sys::cuDeviceSetMemPool(dev, previous)
                .result()
                .map_err(cuda_err("cuDeviceSetMemPool"))?;
        }
        out
    }

    /// Hand every block no live buffer uses back to the driver. Synchronizes
    /// first, so frees still queued on the stream have landed.
    pub fn trim(&self) -> Result<()> {
        self.device.synchronize()?;
        self.ctx
            .bind_to_thread()
            .map_err(cuda_err("bind context"))?;
        unsafe {
            candle_core::cuda_backend::cudarc::driver::sys::cuMemPoolTrimTo(self.pool, 0)
                .result()
                .map_err(cuda_err("cuMemPoolTrimTo"))
        }
    }

    /// `(reserved, used)` bytes — what the pool holds from the driver, and how
    /// much of it live buffers occupy.
    pub fn usage(&self) -> Option<(usize, usize)> {
        pool_usage(&self.ctx, self.pool)
    }
}

#[cfg(feature = "cuda")]
impl Drop for DevicePool {
    fn drop(&mut self) {
        // Deferred by the driver until the last buffer from it is freed.
        if self.ctx.bind_to_thread().is_ok() {
            unsafe {
                let _ = candle_core::cuda_backend::cudarc::driver::sys::cuMemPoolDestroy(self.pool);
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_err(
    what: &'static str,
) -> impl Fn(candle_core::cuda_backend::cudarc::driver::DriverError) -> candle_core::Error {
    move |e| candle_core::Error::Msg(format!("{what}: {e}"))
}

#[cfg(feature = "cuda")]
fn pool_usage(
    ctx: &candle_core::cuda_backend::cudarc::driver::CudaContext,
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
) -> Option<(usize, usize)> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    ctx.bind_to_thread().ok()?;
    let attr = |a| unsafe {
        let mut v: u64 = 0;
        sys::cuMemPoolGetAttribute(pool, a, &mut v as *mut u64 as *mut _)
            .result()
            .ok()
            .map(|_| v as usize)
    };
    Some((
        attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
        attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
    ))
}

/// Without CUDA there is no pool to own: [`DevicePool::new`] always answers
/// `None`, so no value of this type exists.
#[cfg(not(feature = "cuda"))]
pub enum DevicePool {}

#[cfg(not(feature = "cuda"))]
impl DevicePool {
    pub fn new(_device: &Device) -> Result<Option<Self>> {
        Ok(None)
    }
    pub fn scope<T>(&self, _f: impl FnOnce() -> Result<T>) -> Result<T> {
        match *self {}
    }
    pub fn trim(&self) -> Result<()> {
        match *self {}
    }
    pub fn usage(&self) -> Option<(usize, usize)> {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::Tensor;

    #[test]
    fn reserve_and_release_round_trip() {
        let ledger = VramLedger::new(1000, None);
        let a = ledger.reserve(400, "a").unwrap();
        let b = ledger.reserve(600, "b").unwrap();
        assert_eq!(ledger.reserved(), 1000);
        drop(a);
        assert_eq!(ledger.reserved(), 600);
        drop(b);
        assert_eq!(ledger.reserved(), 0);
        assert_eq!(ledger.high_water(), 1000);
    }

    #[test]
    fn a_reservation_that_does_not_fit_is_refused_and_books_nothing() {
        let ledger = VramLedger::new(1000, None);
        let _a = ledger.reserve(700, "a").unwrap();
        let err = ledger.reserve(400, "kv cache").unwrap_err();
        assert!(is_vram_exhausted(&err), "{err}");
        assert!(err.to_string().contains("kv cache"), "{err}");
        assert_eq!(ledger.reserved(), 700);
    }

    #[test]
    fn resize_grows_and_shrinks_in_place() {
        let ledger = VramLedger::new(1000, None);
        let mut r = ledger.reserve(100, "r").unwrap();
        r.resize(900, "r").unwrap();
        assert_eq!(ledger.reserved(), 900);
        assert!(r.resize(1100, "r").is_err());
        assert_eq!((ledger.reserved(), r.bytes()), (900, 900));
        r.resize(50, "r").unwrap();
        assert_eq!(ledger.reserved(), 50);
        drop(r);
        assert_eq!(ledger.reserved(), 0);
    }

    fn tiny_qtensor() -> Result<QTensor> {
        let w = Tensor::from_vec(vec![0.1f32; 32 * 32], (32, 32), &Device::Cpu)?;
        QTensor::quantize(&w, GgmlDType::Q4_0)
    }

    /// The expert cache is the shock absorber: it fills what nothing has
    /// reserved, yields it the moment a reservation needs it, and grows back
    /// when the reservation is released.
    #[test]
    fn the_expert_cache_yields_to_reservations_and_grows_back() {
        let cache = ExpertCache::elastic(None);
        let ledger = VramLedger::new(1000, Some(cache.clone()));
        assert_eq!(cache.budget_bytes(), 1000);
        for i in 0..10 {
            cache.get((0, i), 100, tiny_qtensor).unwrap();
        }
        assert_eq!(cache.stats().resident_bytes, 1000);

        let kv = ledger.reserve(650, "kv cache").unwrap();
        assert_eq!(cache.budget_bytes(), 350);
        assert_eq!(
            cache.stats().resident_bytes,
            300,
            "evicted down to the budget"
        );
        assert_eq!(cache.stats().evicted_bytes, 700);

        drop(kv);
        assert_eq!(cache.budget_bytes(), 1000);
        for i in 10..17 {
            cache.get((0, i), 100, tiny_qtensor).unwrap();
        }
        assert_eq!(
            cache.stats().resident_bytes,
            1000,
            "grew back into the room"
        );
    }

    /// A refusal happens only when the reservation cannot fit with the cache
    /// *empty* — a full cache is never a reason to refuse — and a refused
    /// reservation leaves the cache alone.
    #[test]
    fn a_full_cache_never_causes_a_refusal() {
        let cache = ExpertCache::elastic(None);
        let ledger = VramLedger::new(1000, Some(cache.clone()));
        for i in 0..10 {
            cache.get((0, i), 100, tiny_qtensor).unwrap();
        }
        let _kv = ledger.reserve(1000, "kv cache").unwrap();
        assert_eq!(cache.stats().resident_bytes, 0);
        assert!(ledger.reserve(1, "more").is_err());
    }

    /// A transient no larger than one already seen leaves the cache alone —
    /// the headroom for it was taken the first time — and only a new high
    /// shrinks it further. Refusal still looks at what is live.
    #[test]
    fn transients_are_held_back_once_not_per_forward() {
        let cache = ExpertCache::elastic(None);
        let ledger = VramLedger::new(1000, Some(cache.clone()));
        drop(ledger.reserve_transient(200, "prefill").unwrap());
        assert_eq!(cache.budget_bytes(), 800, "headroom kept after release");
        drop(ledger.reserve_transient(50, "decode").unwrap());
        assert_eq!(
            cache.budget_bytes(),
            800,
            "a smaller transient moves nothing"
        );
        let _kv = ledger.reserve(700, "kv cache").unwrap();
        assert_eq!(cache.budget_bytes(), 100);
        assert!(
            ledger.reserve_transient(300, "prefill").is_ok(),
            "300 of 300 left"
        );
        assert!(ledger.reserve_transient(301, "prefill").is_err());
    }

    #[test]
    fn an_explicit_cap_bounds_the_elastic_budget() {
        let cache = ExpertCache::elastic(Some(300));
        let ledger = VramLedger::new(1000, Some(cache.clone()));
        assert_eq!(cache.budget_bytes(), 300);
        let _r = ledger.reserve(800, "kv").unwrap();
        assert_eq!(cache.budget_bytes(), 200);
    }
}

/// What the device actually charges for buffers of a given size — measured,
/// because the ledger's counts are only as good as this. Run on a CUDA box:
/// `cargo test -p gallium-core --features cuda --release -- --ignored
/// device_charge_per_buffer --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod device_charge {
    use super::*;
    use candle_core::{DType, Tensor};

    #[test]
    #[ignore = "needs a CUDA device"]
    fn device_charge_per_buffer() {
        let device = Device::new_cuda(0).unwrap();
        for bytes in [
            64 << 10,
            512 << 10,
            1 << 20,
            1_120_000,
            2 << 20,
            2_230_000,
            4 << 20,
            8_500_000,
            40 << 20,
        ] {
            let n = 64;
            let before = free_device_memory(&device).unwrap().unwrap();
            let held: Vec<Tensor> = (0..n)
                .map(|_| Tensor::zeros(bytes, DType::U8, &device).unwrap())
                .collect();
            let after = free_device_memory(&device).unwrap().unwrap();
            let per = (before - after) as f64 / n as f64;
            println!(
                "{:>10} B buffer: {:>12.0} B charged each ({:.2}x)",
                bytes,
                per,
                per / bytes as f64
            );
            drop(held);
        }
    }
}
