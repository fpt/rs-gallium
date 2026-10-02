//! Overlapped weight streaming (issue #343 stage 5): upload the next streamed
//! weight while the current one is being computed with.
//!
//! A dense model that does not fit the card streams most of its weights every
//! forward ([`crate::quantized::QLinear`]'s streamed form). Uploaded on the
//! compute stream, as candle's `QStorage::from_data` does, a token is the sum of
//! its uploads and its compute — on Qwen3.8-27B, ~140–170 ms of copying and
//! ~40 ms of arithmetic, one after the other. Uploaded on a second stream, the
//! next weight crosses the bus while this one is multiplied, and a token costs
//! roughly the larger of the two.
//!
//! Three pieces:
//!
//! - **Slots.** A few `QTensor`s per (dtype, shape), allocated once, whose bytes
//!   are rewritten in place (`QTensor::device_ptr`) rather than a fresh buffer
//!   allocated per weight per forward. candle cannot build quantized storage
//!   around a buffer of ours, but it does say where its own is.
//! - **A copy stream and two events per slot.** `ready` is recorded on the copy
//!   stream after a slot is filled, and the compute stream waits on it before
//!   using the slot; `released` is recorded on the compute stream once the
//!   kernels reading a slot have been enqueued, and the copy stream waits on it
//!   before overwriting the slot. Both are raw driver calls on a raw stream:
//!   creating a stream through cudarc puts the whole context into its
//!   multi-stream mode, which records and waits on events around every buffer
//!   every kernel touches.
//! - **The order.** A dense model reaches its streamed weights in the same
//!   sequence every forward, so each request records which weight followed the
//!   previous one, and serving a weight starts the upload of the one that
//!   followed it last time.
//!
//! The copies read straight from the GGUF mmap, which `QVarBuilder` registers as
//! page-locked when it streams ([`crate::quantized::QVarBuilder::with_weight_streaming`]):
//! a copy from pageable memory is staged by the driver and cannot overlap.

use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{Device, Result};

/// What the streamer needs to fetch one weight: where its bytes are, and how to
/// rebuild it as a `QTensor` the first time a slot of its kind is made.
#[derive(Clone)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct WeightRef {
    pub key: u64,
    pub source: Arc<crate::quantized::MmapSource>,
    /// Byte range of the weight in `source.mmap`.
    pub start: usize,
    pub len: usize,
    pub class: (candle_core::quantized::GgmlDType, Vec<usize>),
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl WeightRef {
    fn bytes(&self) -> &[u8] {
        &self.source.mmap[self.start..self.start + self.len]
    }

    /// The weight as a fresh `QTensor` on `device` — how a slot is first made.
    fn build(&self, device: &Device) -> Result<QTensor> {
        let storage = candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Borrowed(self.bytes()),
            device,
            self.class.0,
        )?;
        QTensor::new(storage, self.class.1.clone())
    }
}

#[cfg(feature = "cuda")]
pub(crate) use imp::WeightStreamer;

#[cfg(not(feature = "cuda"))]
pub(crate) enum WeightStreamer {}

#[cfg(not(feature = "cuda"))]
impl WeightStreamer {
    pub fn new(_device: &Device) -> Result<Option<Self>> {
        Ok(None)
    }
    pub fn request(&self, _w: &WeightRef) -> Result<Arc<QTensor>> {
        match *self {}
    }
}

#[cfg(feature = "cuda")]
mod imp {
    use super::*;
    use candle_core::cuda_backend::cudarc::driver::{sys, CudaContext};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Slots per (dtype, shape): the one being computed with and the one being
    /// filled for the next request of that kind.
    const SLOTS_PER_CLASS: usize = 2;

    struct Slot {
        qt: Arc<QTensor>,
        /// The weight whose bytes the slot holds (or is being filled with).
        holds: Option<u64>,
        ready: sys::CUevent,
        released: sys::CUevent,
    }

    struct Inner {
        classes: HashMap<(candle_core::quantized::GgmlDType, Vec<usize>), Vec<Slot>>,
        /// For each weight, the one requested right after it last time.
        next_after: HashMap<u64, WeightRef>,
        prev: Option<u64>,
        /// Slots handed out since the last request: their `released` is
        /// recorded at the next one, after their kernels were enqueued.
        in_use: Vec<((candle_core::quantized::GgmlDType, Vec<usize>), usize)>,
    }

    pub(crate) struct WeightStreamer {
        ctx: Arc<CudaContext>,
        compute: sys::CUstream,
        copy: sys::CUstream,
        /// Where slots are allocated — their own pool, so a forward's
        /// `TransientProbe` on the default pool does not count them as scratch.
        pool: crate::vram::DevicePool,
        device: Device,
        inner: Mutex<Inner>,
    }

    // Raw driver handles, used only under `inner`'s lock with the context bound.
    unsafe impl Send for WeightStreamer {}
    unsafe impl Sync for WeightStreamer {}

    fn check(what: &'static str, r: sys::CUresult) -> Result<()> {
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            candle_core::bail!("{what}: {r:?}")
        }
    }

    impl WeightStreamer {
        pub fn new(device: &Device) -> Result<Option<Self>> {
            let Device::Cuda(cuda) = device else {
                return Ok(None);
            };
            let Some(pool) = crate::vram::DevicePool::new(device)? else {
                return Ok(None);
            };
            let stream = cuda.cuda_stream();
            let ctx = stream.context().clone();
            ctx.bind_to_thread()
                .map_err(|e| candle_core::Error::Msg(format!("bind: {e}")))?;
            let mut copy = std::ptr::null_mut();
            check("cuStreamCreate", unsafe {
                sys::cuStreamCreate(
                    &mut copy,
                    sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
                )
            })?;
            Ok(Some(Self {
                ctx,
                compute: stream.cu_stream(),
                copy,
                pool,
                device: device.clone(),
                inner: Mutex::new(Inner {
                    classes: HashMap::new(),
                    next_after: HashMap::new(),
                    prev: None,
                    in_use: Vec::new(),
                }),
            }))
        }

        fn new_event() -> Result<sys::CUevent> {
            let mut ev = std::ptr::null_mut();
            check("cuEventCreate", unsafe {
                sys::cuEventCreate(&mut ev, sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32)
            })?;
            Ok(ev)
        }

        /// Fill `slot` with `w`'s bytes on the copy stream, once whatever last
        /// read it is done.
        fn fill(&self, slot: &mut Slot, w: &WeightRef) -> Result<()> {
            let src = w.bytes();
            unsafe {
                check(
                    "cuStreamWaitEvent",
                    sys::cuStreamWaitEvent(self.copy, slot.released, 0),
                )?;
                check(
                    "cuMemcpyHtoDAsync",
                    sys::cuMemcpyHtoDAsync_v2(
                        slot.qt.device_ptr()? as sys::CUdeviceptr,
                        src.as_ptr() as *const _,
                        src.len(),
                        self.copy,
                    ),
                )?;
                check("cuEventRecord", sys::cuEventRecord(slot.ready, self.copy))?;
            }
            slot.holds = Some(w.key);
            Ok(())
        }

        /// Make sure `class` has its slots, building them from `w` (whose bytes
        /// the first one then already holds).
        fn ensure_class(&self, inner: &mut Inner, w: &WeightRef) -> Result<()> {
            if inner.classes.contains_key(&w.class) {
                return Ok(());
            }
            let mut slots = Vec::with_capacity(SLOTS_PER_CLASS);
            for i in 0..SLOTS_PER_CLASS {
                let qt = Arc::new(self.pool.scope(|| w.build(&self.device))?);
                slots.push(Slot {
                    qt,
                    holds: (i == 0).then_some(w.key),
                    ready: Self::new_event()?,
                    released: Self::new_event()?,
                });
            }
            // The first slot was filled synchronously on the compute stream by
            // `build`; its `ready` is recorded there so waiting on it is a no-op.
            unsafe {
                check(
                    "cuEventRecord",
                    sys::cuEventRecord(slots[0].ready, self.compute),
                )?
            };
            inner.classes.insert(w.class.clone(), slots);
            Ok(())
        }

        /// Index of the slot in `class` to fill next: not `avoid`, and the
        /// least recently handed out otherwise (slots alternate).
        fn victim(slots: &[Slot], avoid: Option<usize>, keep: Option<u64>) -> usize {
            (0..slots.len())
                .find(|&i| Some(i) != avoid && (keep.is_none() || slots[i].holds != keep))
                .unwrap_or(0)
        }

        /// `w` as a `QTensor` on the device, ready on the compute stream; and
        /// the upload of whatever weight followed `w` last time, started on the
        /// copy stream.
        ///
        /// The contract, which nothing checks: every kernel reading the
        /// returned tensor is enqueued before the next `request`, and the
        /// tensor is not kept past it. The next request marks the slot free
        /// and a later one overwrites it in place, so a tensor held across
        /// requests silently computes with another weight's bytes.
        /// `QLinear::forward` requests, multiplies and drops it in one call.
        pub fn request(&self, w: &WeightRef) -> Result<Arc<QTensor>> {
            self.ctx
                .bind_to_thread()
                .map_err(|e| candle_core::Error::Msg(format!("bind: {e}")))?;
            let mut inner = self.inner.lock().unwrap();

            // Everything handed out before this request has had its kernels
            // enqueued by now: mark the point after which its slot is free.
            for (class, i) in std::mem::take(&mut inner.in_use) {
                if let Some(slot) = inner.classes.get(&class).and_then(|s| s.get(i)) {
                    unsafe {
                        check(
                            "cuEventRecord",
                            sys::cuEventRecord(slot.released, self.compute),
                        )?
                    };
                }
            }

            if let Some(prev) = inner.prev.replace(w.key) {
                inner.next_after.insert(prev, w.clone());
            }

            self.ensure_class(&mut inner, w)?;
            let next = inner.next_after.get(&w.key).cloned();

            let slots = inner.classes.get_mut(&w.class).expect("ensured above");
            let current = match slots.iter().position(|s| s.holds == Some(w.key)) {
                Some(i) => i,
                None => {
                    let i = Self::victim(slots, None, next.as_ref().map(|n| n.key));
                    self.fill(&mut slots[i], w)?;
                    i
                }
            };
            unsafe {
                check(
                    "cuStreamWaitEvent",
                    sys::cuStreamWaitEvent(self.compute, slots[current].ready, 0),
                )?
            };
            let qt = slots[current].qt.clone();
            inner.in_use.push((w.class.clone(), current));

            // Start the next weight's upload while this one is computed with.
            if let Some(n) = next.filter(|n| n.key != w.key) {
                self.ensure_class(&mut inner, &n)?;
                let same_class = n.class == w.class;
                let slots = inner.classes.get_mut(&n.class).expect("ensured above");
                if !slots.iter().any(|s| s.holds == Some(n.key)) {
                    let avoid = same_class.then_some(current);
                    let i = Self::victim(slots, avoid, None);
                    self.fill(&mut slots[i], &n)?;
                }
            }
            Ok(qt)
        }
    }

    impl Drop for WeightStreamer {
        fn drop(&mut self) {
            if self.ctx.bind_to_thread().is_err() {
                return;
            }
            unsafe {
                let _ = sys::cuStreamSynchronize(self.copy);
                if let Ok(inner) = self.inner.lock() {
                    for slots in inner.classes.values() {
                        for s in slots {
                            let _ = sys::cuEventDestroy_v2(s.ready);
                            let _ = sys::cuEventDestroy_v2(s.released);
                        }
                    }
                }
                let _ = sys::cuStreamDestroy_v2(self.copy);
            }
        }
    }
}
