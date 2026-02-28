//! Support for CUDA graph capture.
//!
//! During CUDA graph capture, `memcpy_stod` records two nodes in the graph:
//! 1. A `cudaMallocAsync` node (GPU allocation)
//! 2. A `cuMemcpyHtoDAsync` node (host→device copy from the source Vec<usize>)
//!
//! On graph replay, the memcpy node reads from the ORIGINAL host pointer.
//! If the source Vec has been freed, replay reads garbage → kernel crash.
//!
//! This module prevents both deallocations:
//! - GPU buffer: wrapped in `ManuallyDrop` (prevents cudaFreeAsync)
//! - Host Vec: leaked via `std::mem::forget` (prevents heap deallocation)
//!
//! Note: intermediate tensor address reuse (the main source of graph replay
//! corruption) is handled by a dedicated capture memory pool in mistral.rs's
//! `cuda_graph::switch_to_capture_pool()` / `restore_default_pool()`, NOT by
//! tensor retention. This module only handles stride buffer persistence.

use cudarc::driver::CudaSlice;
use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::ops::Deref;

thread_local! {
    static GRAPH_CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// Signal that CUDA graph capture is beginning on this thread.
pub fn begin_graph_capture() {
    GRAPH_CAPTURING.with(|c| c.set(true));
}

/// Signal that CUDA graph capture has ended on this thread.
pub fn end_graph_capture() {
    GRAPH_CAPTURING.with(|c| c.set(false));
}

/// Returns true if the current thread is inside a CUDA graph capture.
pub fn is_graph_capturing() -> bool {
    GRAPH_CAPTURING.with(|c| c.get())
}

/// Leak a host-side Vec so its heap allocation survives graph replay.
/// The graph's `cuMemcpyHtoDAsync` node holds a pointer to this data.
///
/// Cost: ~64 bytes per stride buffer, ~900 bytes per graph capture total.
pub fn leak_host_data<T>(data: Vec<T>) {
    std::mem::forget(data);
}

/// A GPU buffer that is either normally managed or persistent (leaked) for graph capture.
///
/// When created during graph capture, the underlying `CudaSlice` is wrapped in
/// `ManuallyDrop` so it is never freed — its device pointer remains valid for
/// graph replay. The source host Vec must also be leaked separately via
/// `leak_host_data()`.
pub enum MaybePersistent<T> {
    Normal(CudaSlice<T>),
    Persistent(ManuallyDrop<CudaSlice<T>>),
}

impl<T> MaybePersistent<T> {
    /// Wrap a `CudaSlice` and leak the source host data if capturing a graph.
    ///
    /// The `source` Vec is the host data that was passed to `memcpy_stod`.
    /// During graph capture, the graph records a memcpy node that reads from
    /// this Vec's heap pointer. We leak it so replay can re-read from it.
    pub fn new(slice: CudaSlice<T>, source: Vec<T>) -> Self {
        if is_graph_capturing() {
            leak_host_data(source);
            MaybePersistent::Persistent(ManuallyDrop::new(slice))
        } else {
            MaybePersistent::Normal(slice)
        }
    }
}

impl<T> Deref for MaybePersistent<T> {
    type Target = CudaSlice<T>;

    fn deref(&self) -> &CudaSlice<T> {
        match self {
            MaybePersistent::Normal(s) => s,
            MaybePersistent::Persistent(s) => s,
        }
    }
}
