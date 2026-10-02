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

use candle_core::{Device, Result, Tensor};

use crate::ExpertCache;

/// The device-memory budget for one loaded model, and what has been reserved
/// against it. Shared (`Arc`) by everything that allocates on the device.
pub struct VramLedger {
    /// The budget before any recalibration: free memory at load less the margin.
    base_budget: usize,
    /// Driver-reported free memory when the ledger was made — what
    /// [`Self::recalibrate`] measures later readings against. `None` for a
    /// ledger given a plain budget (tests), which never recalibrates.
    free_at_load: Option<usize>,
    state: Mutex<LedgerState>,
    expert_cache: Option<Arc<ExpertCache>>,
}

struct LedgerState {
    /// What the ledger allows: `base_budget`, less whatever the driver has
    /// shown is in use that nothing books — see [`VramLedger::recalibrate`].
    budget: usize,
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
    /// The expert cache is held at a budget of zero whatever is reserved —
    /// see [`VramLedger::with_cache_closed`].
    cache_closed: bool,
    /// What a prefill window's model estimate is multiplied by when booked —
    /// the most `measured / estimated` has ever been, never below 1.
    prefill_scale: f64,
    /// The largest prefill window measured, for a model with no estimate to
    /// scale: booked as the window's minimum.
    prefill_floor: usize,
    /// The largest **first** window measured — position 0, where the model
    /// allocates its recurrent states, its first KV buffers and the kernels'
    /// workspaces. Kept apart from `prefill_scale`, which it would otherwise
    /// inflate for every later window: on Qwen3.8-27B the first window costs
    /// ~3x its estimate and the rest ~1x.
    first_window_floor: usize,
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
        Self::build(budget, None, expert_cache)
    }

    /// A ledger over `free_at_load - margin`, which remembers `free_at_load` so
    /// that [`Self::recalibrate`] can correct it from later driver readings.
    pub fn calibrated(
        free_at_load: usize,
        margin: usize,
        expert_cache: Option<Arc<ExpertCache>>,
    ) -> Arc<Self> {
        Self::build(
            free_at_load.saturating_sub(margin),
            Some(free_at_load),
            expert_cache,
        )
    }

    fn build(
        budget: usize,
        free_at_load: Option<usize>,
        expert_cache: Option<Arc<ExpertCache>>,
    ) -> Arc<Self> {
        if let Some(cache) = &expert_cache {
            cache.set_budget(budget);
        }
        Arc::new(Self {
            base_budget: budget,
            free_at_load,
            state: Mutex::new(LedgerState {
                budget,
                persistent: 0,
                transient: 0,
                headroom: 0,
                high_water: 0,
                cache_closed: false,
                prefill_scale: 1.0,
                prefill_floor: 0,
                first_window_floor: 0,
            }),
            expert_cache,
        })
    }

    pub fn budget(&self) -> usize {
        self.state.lock().unwrap().budget
    }

    /// Correct the budget from a driver reading taken between calls (no
    /// transient live), and return `(unbooked, budget)`.
    ///
    /// Not everything on the device can be booked where it is allocated —
    /// cuBLAS and flash-attn workspaces, a model's lazily-copied bias tables,
    /// slack the memory pools hold — and the fixed margin the budget was cut
    /// with is a guess at all of it. A guess that was right for GPT-OSS 20B was
    /// ~670 MiB short for 120B: the margin was gone after one call, and the
    /// next prefill's booked transient OOM'd the card. So the driver is asked:
    /// whatever it shows in use since load beyond what is booked — reservations
    /// plus what the expert cache's pools hold — is taken off the budget, which
    /// keeps the margin a margin. It can give it back too, when that use falls.
    pub fn recalibrate(&self, driver_free: usize) -> Option<(usize, usize)> {
        let free_at_load = self.free_at_load?;
        // Read before the ledger lock: the cache is only ever locked after it.
        let cache_footprint = self.expert_cache.as_ref().map_or(0, |c| c.footprint());
        let mut st = self.state.lock().unwrap();
        let used = free_at_load.saturating_sub(driver_free);
        let unbooked = used.saturating_sub(st.live() + cache_footprint);
        let budget = self.base_budget.saturating_sub(unbooked);
        if budget != st.budget {
            st.budget = budget;
            self.sync_cache(&st);
        }
        Some((unbooked, budget))
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

    /// Bytes held by persistent reservations (KV and its checkpoints).
    pub fn persistent(&self) -> usize {
        self.state.lock().unwrap().persistent
    }

    /// A forward over `tokens` tokens, for which the model estimated
    /// `estimated` bytes of scratch, was measured to need `measured`
    /// ([`TransientProbe`]). A model's `transient_bytes` is an estimate; this is
    /// what it actually used — on Qwen3.8-27B a first prefill window peaked at
    /// ~1.5 GB against an estimate of nothing, and the cache filling into that
    /// room OOM'd the card.
    ///
    /// Learned two ways, because the two kinds of forward differ by two orders
    /// of magnitude. A **decode** step's scratch is small and the same every
    /// token, so it becomes headroom held back from the cache for good — paying
    /// for it per step would evict a few experts every token. A **prefill**
    /// window's grows with the context, so holding its peak back for good cost
    /// a 24k-token prompt's decode its whole cache (5.7 GB held back, 1.6 tok/s);
    /// instead it corrects the estimate — the window books
    /// [`Self::prefill_booking`] while it runs, the cache gives that room up for
    /// it and takes it back after, and a correction learned at one length still
    /// scales with the next.
    pub fn observe_transient(&self, tokens: usize, pos: usize, estimated: usize, measured: usize) {
        let mut st = self.state.lock().unwrap();
        if tokens <= 1 {
            if measured > st.headroom {
                st.headroom = measured;
                self.sync_cache(&st);
            }
            return;
        }
        if pos == 0 {
            st.first_window_floor = st.first_window_floor.max(measured);
            return;
        }
        if estimated > 0 {
            st.prefill_scale = st.prefill_scale.max(measured as f64 / estimated as f64);
        }
        st.prefill_floor = st.prefill_floor.max(measured);
    }

    /// What a prefill window at `pos`, which the model estimates at
    /// `estimated` bytes, is booked at: the estimate times the learned
    /// correction — or, for a model with no estimate, the largest window
    /// measured — and never below the measured first window when it is one.
    pub fn prefill_booking(&self, pos: usize, estimated: usize) -> usize {
        let st = self.state.lock().unwrap();
        let scaled = if estimated == 0 {
            st.prefill_floor
        } else {
            (estimated as f64 * st.prefill_scale).ceil() as usize
        };
        if pos == 0 {
            scaled.max(st.first_window_floor)
        } else {
            scaled
        }
    }

    /// Run `f` with the expert cache's budget at zero — nothing admitted, every
    /// streamed weight passing through — then give it back. For a calibration
    /// forward, whose scratch has to be measured before the cache fills the
    /// room it needs.
    ///
    /// Closed in the ledger's state, not by setting the cache's budget once:
    /// every reservation re-syncs the cache, and the forward's own transient
    /// reservation would otherwise reopen it the moment the forward started.
    pub fn with_cache_closed<T>(&self, f: impl FnOnce() -> T) -> T {
        {
            let mut st = self.state.lock().unwrap();
            st.cache_closed = true;
            self.sync_cache(&st);
        }
        let out = f();
        let mut st = self.state.lock().unwrap();
        st.cache_closed = false;
        self.sync_cache(&st);
        out
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
        let available = st.budget.saturating_sub(st.live());
        if bytes > available {
            return Err(candle_core::Error::wrap(VramExhausted {
                what,
                wanted: bytes,
                available,
                budget: st.budget,
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
            if st.cache_closed {
                cache.set_budget(0);
                return;
            }
            let held_back = st.persistent + st.headroom.max(st.transient);
            cache.set_budget(st.budget.saturating_sub(held_back));
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

/// What the plain allocator (`cuMemAlloc`) rounds every buffer up to — 2 MiB
/// on the 4070 (`plain_alloc_frees`: 200 buffers of 4.4 MB took 1200 MiB).
const PLAIN_ALLOC_GRANULARITY: usize = 2 << 20;

/// What a [`upload_plain`] buffer of `bytes` costs `device`: rounded to the
/// plain allocator's granularity on CUDA, as-is anywhere else.
pub fn plain_charge(bytes: usize, device: &Device) -> usize {
    if device.is_cuda() {
        bytes.div_ceil(PLAIN_ALLOC_GRANULARITY) * PLAIN_ALLOC_GRANULARITY
    } else {
        bytes
    }
}

/// `parts`, concatenated, as one `U8` tensor on `device` — on CUDA in a buffer
/// from the **plain allocator** (`cuMemAlloc`), not a memory pool.
///
/// For the expert cache's MXFP4 entries (issue #343 stage 4). A pool returns
/// memory to the driver only in whole chunks (~32 MB), and an LRU frees
/// entries out of allocation order, so it never frees a whole chunk: measured
/// (`pool_trim_granularity`), freeing every other entry and trimming returned
/// 0% — padded to 2 MiB multiples or not — and a cache shrinking to make room
/// for KV evicted nearly everything before the pool gave anything back. A plain
/// allocation goes back the moment it is freed (`plain_alloc_frees`), and the
/// cudarc slice that owns it frees it with `cuMemFreeAsync`, which accepts it.
/// The cost is the 2 MiB rounding ([`plain_charge`]), which is why an entry is
/// a whole expert (13.2 MB → 14 MiB) rather than one matrix (4.4 MB → 6 MiB).
pub fn upload_plain(device: &Device, parts: &[&[u8]]) -> Result<Tensor> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda) => {
            use candle_core::cuda_backend::cudarc::driver::sys;
            let stream = cuda.cuda_stream();
            stream
                .context()
                .bind_to_thread()
                .map_err(cuda_err("bind context"))?;
            let mut ptr: sys::CUdeviceptr = 0;
            unsafe {
                sys::cuMemAlloc_v2(&mut ptr, total.max(1))
                    .result()
                    .map_err(cuda_err("cuMemAlloc"))?;
            }
            // SAFETY: `ptr` is a live allocation of `total` bytes, owned from
            // here on by the slice, which frees it on drop.
            let mut slice = unsafe { stream.upgrade_device_ptr::<u8>(ptr, total) };
            let mut at = 0;
            for part in parts {
                stream
                    .memcpy_htod(*part, &mut slice.slice_mut(at..at + part.len()))
                    .map_err(cuda_err("memcpy_htod"))?;
                at += part.len();
            }
            let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, cuda.clone());
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                total,
                candle_core::op::BackpropOp::none(),
                false,
            ))
        }
        _ => Tensor::from_vec(parts.concat(), total, device),
    }
}

/// The last CUDA error cudarc swallowed since the previous call, if any.
/// cudarc records — and otherwise drops — failures in paths with nowhere to
/// return them, `Drop` above all: a failed free is a silent leak.
pub fn take_swallowed_error(device: &Device) -> Option<String> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(cuda) => cuda
            .cuda_stream()
            .context()
            .check_err()
            .err()
            .map(|e| e.to_string()),
        _ => {
            let _ = device;
            None
        }
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

/// Hand the device's default pool's unused chunks back to the driver, after a
/// synchronize. A prefill window's scratch grows with its position, so each
/// window's buffers are a little larger than the last one's and do not fit the
/// blocks it freed: without a trim the pool keeps every size it has ever
/// needed. Measured on Qwen3.8-27B: the pool held 5.0 GB for 2.5 GB in use by
/// a 4k-token prefill's sixth window, and the card ran out.
pub fn trim_default_pool(device: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::sys;
        let Device::Cuda(cuda) = device else {
            return Ok(());
        };
        device.synchronize()?;
        let ctx = cuda.cuda_stream().context().clone();
        ctx.bind_to_thread().map_err(cuda_err("bind context"))?;
        let mut pool = std::ptr::null_mut();
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device())
                .result()
                .map_err(cuda_err("cuDeviceGetDefaultMemPool"))?;
            sys::cuMemPoolTrimTo(pool, 0)
                .result()
                .map_err(cuda_err("cuMemPoolTrimTo"))?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}

/// Measures what one forward allocates at its peak from the device's default
/// pool — where every activation, scratch buffer, and uploaded-for-one-forward
/// weight lands — by resetting the pool's high-water mark first and reading it
/// after. Exact for any model, which a hand-written `transient_bytes` is not.
pub struct TransientProbe {
    #[cfg(feature = "cuda")]
    pool: Option<(
        Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
        candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
        usize,
        usize,
    )>,
}

impl TransientProbe {
    /// Start measuring on `device`. A no-op off CUDA.
    pub fn start(device: &Device) -> Self {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::sys;
            let pool = (|| {
                let Device::Cuda(cuda) = device else {
                    return None;
                };
                let ctx = cuda.cuda_stream().context().clone();
                ctx.bind_to_thread().ok()?;
                let mut pool = std::ptr::null_mut();
                unsafe { sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) }
                    .result()
                    .ok()?;
                let (reserved, used) = pool_usage(&ctx, pool)?;
                for attr in [
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                ] {
                    let mut zero: u64 = 0;
                    unsafe {
                        sys::cuMemPoolSetAttribute(pool, attr, &mut zero as *mut u64 as *mut _)
                    }
                    .result()
                    .ok()?;
                }
                Some((ctx, pool, used, reserved))
            })();
            Self { pool }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            Self {}
        }
    }

    /// What the forward cost the device since [`Self::start`]: the larger of
    /// its peak bytes in use and how far it grew what the pool holds from the
    /// driver. The second is what fragmentation costs — measured on
    /// Qwen3.8-27B, a first window's used peak was ~1.5 GB while the pool's
    /// reservation grew ~1.9 GB, and chunks a pool reserved are not handed back
    /// while any of it is live.
    pub fn peak(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::sys;
            let (ctx, pool, used0, reserved0) = self.pool.as_ref()?;
            ctx.bind_to_thread().ok()?;
            let read = |attr| {
                let mut v: u64 = 0;
                unsafe { sys::cuMemPoolGetAttribute(*pool, attr, &mut v as *mut u64 as *mut _) }
                    .result()
                    .ok()
                    .map(|_| v as usize)
            };
            let used = read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?;
            let reserved = read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?;
            Some(
                used.saturating_sub(*used0)
                    .max(reserved.saturating_sub(*reserved0)),
            )
        }
        #[cfg(not(feature = "cuda"))]
        None
    }
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

    /// Memory in use that nothing booked comes off the budget — and so off the
    /// expert cache — once the driver shows it, and goes back when it is gone.
    #[test]
    fn recalibration_takes_unbooked_use_off_the_budget() {
        let cache = ExpertCache::elastic(None);
        let ledger = VramLedger::calibrated(10_000, 1_000, Some(cache.clone()));
        assert_eq!(ledger.budget(), 9_000);
        let _kv = ledger.reserve(2_000, "kv").unwrap();
        // The driver shows 2 500 in use: the 2 000 booked plus 500 nobody did.
        assert_eq!(ledger.recalibrate(7_500), Some((500, 8_500)));
        assert_eq!(cache.budget_bytes(), 6_500, "the cache gave the 500 up");
        // A reading with everything accounted for restores the budget.
        assert_eq!(ledger.recalibrate(8_000), Some((0, 9_000)));
        assert_eq!(
            VramLedger::new(10, None).recalibrate(5),
            None,
            "no baseline"
        );
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

/// How much a pool gives back when its entries are freed out of allocation
/// order — what decides whether evicting an LRU expert frees driver memory.
/// `cargo test -p gallium-core --features cuda --release -- --ignored
/// pool_trim_granularity --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod pool_trim {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn pool_trim_granularity() {
        let dev = Device::new_cuda(0).unwrap();
        for &(name, bytes) in &[
            ("4.4 MB entries", 4_406_400usize),
            ("padded to 6 MiB", 6 << 20),
            ("2.2 MB entries", 2_203_200),
            ("padded to 4 MiB", 4 << 20),
        ] {
            let pool = DevicePool::new(&dev).unwrap().unwrap();
            let host = vec![1u8; bytes];
            let mut held: Vec<Option<Tensor>> = (0..200)
                .map(|_| {
                    Some(
                        pool.scope(|| Tensor::from_slice(&host, bytes, &dev))
                            .unwrap(),
                    )
                })
                .collect();
            pool.trim().unwrap();
            let (r0, u0) = pool.usage().unwrap();
            for t in held.iter_mut().step_by(2) {
                *t = None;
            }
            pool.trim().unwrap();
            let (r1, u1) = pool.usage().unwrap();
            println!(
                "{name:>16}: full {:>4}/{:>4} MiB; every other freed + trimmed: {:>4}/{:>4} MiB \
                 (reserved/used) — {:.0}% of the freed bytes returned",
                r0 >> 20,
                u0 >> 20,
                r1 >> 20,
                u1 >> 20,
                100.0 * (r0 - r1) as f64 / (u0 - u1) as f64
            );
        }
    }
}

/// Whether memory from the plain allocator (`cuMemAlloc`) can be owned by a
/// cudarc slice, whose drop frees it with `cuMemFreeAsync`, and comes back
/// to the driver when it does. `cargo test -p gallium-core --features cuda
/// --release -- --ignored plain_alloc_frees --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod plain_alloc {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::sys;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn plain_alloc_frees() {
        let dev = Device::new_cuda(0).unwrap();
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        let ctx = stream.context().clone();
        ctx.bind_to_thread().unwrap();
        let free0 = free_device_memory(&dev).unwrap().unwrap();
        let bytes = 4_406_400usize;
        let mut held = Vec::new();
        for _ in 0..200 {
            let mut ptr: sys::CUdeviceptr = 0;
            unsafe { sys::cuMemAlloc_v2(&mut ptr, bytes).result().unwrap() };
            held.push(Some(unsafe { stream.upgrade_device_ptr::<u8>(ptr, bytes) }));
        }
        let free1 = free_device_memory(&dev).unwrap().unwrap();
        for s in held.iter_mut().step_by(2) {
            *s = None;
        }
        let free2 = free_device_memory(&dev).unwrap().unwrap();
        println!(
            "allocated {} MiB; freed every other: {} MiB came back; swallowed error: {:?}",
            (free0 - free1) >> 20,
            (free2 - free1) >> 20,
            ctx.check_err()
        );
    }
}

/// Host→device copy bandwidth on this machine: pageable memory (what a
/// `Vec` or a file mmap is) against page-locked memory. What decides whether
/// streaming weights that do not fit the card is viable. `cargo test -p
/// gallium-core --features cuda --release -- --ignored h2d_bandwidth --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod h2d_bandwidth {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn h2d_bandwidth() {
        let dev = Device::new_cuda(0).unwrap();
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        let ctx = stream.context().clone();
        let bytes = 512usize << 20;
        let mut dst = stream.alloc_zeros::<u8>(bytes).unwrap();
        let time = |label: &str, f: &mut dyn FnMut()| {
            f();
            stream.synchronize().unwrap();
            let n = 6;
            let t = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            stream.synchronize().unwrap();
            let gbs = (bytes * n) as f64 / t.elapsed().as_secs_f64() / 1e9;
            println!("{label:>10}: {gbs:.1} GB/s");
        };
        let pageable = vec![3u8; bytes];
        time("pageable", &mut || {
            stream.memcpy_htod(&pageable, &mut dst).unwrap()
        });
        let mut pinned = unsafe { ctx.alloc_pinned::<u8>(bytes) }.unwrap();
        unsafe { std::ptr::write_bytes(pinned.as_mut_ptr().unwrap(), 3, bytes) };
        time("pinned", &mut || {
            stream.memcpy_htod(&pinned, &mut dst).unwrap()
        });
    }
}

/// What a fresh pool reserves for a single allocation, and whether dropping it
/// and trimming hands it all back — the cost of one pool per cached entry.
/// `cargo test -p gallium-core --features cuda --release -- --ignored
/// pool_per_entry --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod pool_per_entry {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn pool_per_entry() {
        let dev = Device::new_cuda(0).unwrap();
        for &mb in &[4.4f64, 11.0, 23.0, 45.0, 90.0] {
            let bytes = (mb * 1e6) as usize;
            let host = vec![1u8; bytes];
            let free0 = free_device_memory(&dev).unwrap().unwrap();
            let pool = DevicePool::new(&dev).unwrap().unwrap();
            let t = pool
                .scope(|| Tensor::from_slice(&host, bytes, &dev))
                .unwrap();
            let (r, u) = pool.usage().unwrap();
            let free1 = free_device_memory(&dev).unwrap().unwrap();
            drop(t);
            pool.trim().unwrap();
            let (r2, _) = pool.usage().unwrap();
            drop(pool);
            let free2 = free_device_memory(&dev).unwrap().unwrap();
            println!(
                "{mb:>5} MB: pool reserved {:>6.1} MiB for {:>6.1} used, driver charged {:>6.1}; \
                 after drop+trim reserved {:.1}, {:.1} MiB back",
                r as f64 / 1048576.0,
                u as f64 / 1048576.0,
                (free0 - free1) as f64 / 1048576.0,
                r2 as f64 / 1048576.0,
                (free2 as f64 - free1 as f64) / 1048576.0
            );
        }
    }
}
