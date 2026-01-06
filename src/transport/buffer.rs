//! Buffer management for high-performance packet processing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_queue::ArrayQueue;

use crate::MAX_MTU;

/// Default buffer size for packets.
pub const DEFAULT_BUFFER_SIZE: usize = MAX_MTU;

/// A reusable packet buffer.
#[derive(Debug)]
pub struct PacketBuffer {
    data: Vec<u8>,
    len: usize,
}

impl PacketBuffer {
    /// Create a new buffer with default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUFFER_SIZE)
    }

    /// Create a buffer with specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![0u8; capacity],
            len: 0,
        }
    }

    /// Get the buffer data as a slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }

    /// Get the buffer data as a mutable slice (full capacity).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Set the length of valid data.
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.data.len());
        self.len = len;
    }

    /// Get the length of valid data.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Copy data into the buffer.
    pub fn copy_from(&mut self, data: &[u8]) {
        let len = data.len().min(self.data.len());
        self.data[..len].copy_from_slice(&data[..len]);
        self.len = len;
    }

    /// Take ownership of the data.
    pub fn take(self) -> Vec<u8> {
        let mut data = self.data;
        data.truncate(self.len);
        data
    }
}

impl Default for PacketBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<[u8]> for PacketBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PacketBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// Pool of reusable buffers for zero-allocation packet processing.
pub struct BufferPool {
    pool: ArrayQueue<PacketBuffer>,
    buffer_size: usize,
    allocated: AtomicUsize,
    max_buffers: usize,
}

impl BufferPool {
    /// Create a new buffer pool.
    ///
    /// # Arguments
    /// * `initial_count` - Number of buffers to pre-allocate
    /// * `max_count` - Maximum buffers in the pool
    /// * `buffer_size` - Size of each buffer
    pub fn new(initial_count: usize, max_count: usize, buffer_size: usize) -> Arc<Self> {
        let pool = ArrayQueue::new(max_count);

        // Pre-allocate initial buffers
        for _ in 0..initial_count {
            let _ = pool.push(PacketBuffer::with_capacity(buffer_size));
        }

        Arc::new(Self {
            pool,
            buffer_size,
            allocated: AtomicUsize::new(initial_count),
            max_buffers: max_count,
        })
    }

    /// Create with default settings.
    pub fn default_pool() -> Arc<Self> {
        Self::new(64, 1024, DEFAULT_BUFFER_SIZE)
    }

    /// Get a buffer from the pool.
    pub fn get(&self) -> PooledBuffer<'_> {
        let buffer = self.pool.pop().unwrap_or_else(|| {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            PacketBuffer::with_capacity(self.buffer_size)
        });

        PooledBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// Return a buffer to the pool.
    fn return_buffer(&self, mut buffer: PacketBuffer) {
        buffer.clear();
        if self.pool.push(buffer).is_err() {
            // Pool is full, drop the buffer
            self.allocated.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            available: self.pool.len(),
            allocated: self.allocated.load(Ordering::Relaxed),
            max: self.max_buffers,
            buffer_size: self.buffer_size,
        }
    }
}

/// A buffer that automatically returns to its pool when dropped.
pub struct PooledBuffer<'a> {
    buffer: Option<PacketBuffer>,
    pool: &'a BufferPool,
}

impl PooledBuffer<'_> {
    /// Get the inner buffer as a slice.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().unwrap().as_slice()
    }

    /// Get the inner buffer as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().unwrap().as_mut_slice()
    }

    /// Set the length of valid data.
    pub fn set_len(&mut self, len: usize) {
        self.buffer.as_mut().unwrap().set_len(len);
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.as_ref().unwrap().is_empty()
    }

    /// Copy data into the buffer.
    pub fn copy_from(&mut self, data: &[u8]) {
        self.buffer.as_mut().unwrap().copy_from(data);
    }

    /// Take the buffer out of the pool (won't be returned).
    pub fn take(mut self) -> PacketBuffer {
        self.buffer.take().unwrap()
    }
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

impl AsRef<[u8]> for PooledBuffer<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PooledBuffer<'_> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available: usize,
    pub allocated: usize,
    pub max: usize,
    pub buffer_size: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "available={}/{} allocated={} buffer_size={}",
            self.available, self.max, self.allocated, self.buffer_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ring buffer for ordered packet assembly.
    struct PacketRingBuffer {
        buffers: Vec<Option<Vec<u8>>>,
        head: usize,
        base_seq: u64,
        size: usize,
    }

    impl PacketRingBuffer {
        fn new(size: usize) -> Self {
            Self {
                buffers: (0..size).map(|_| None).collect(),
                head: 0,
                base_seq: 0,
                size,
            }
        }

        fn insert(&mut self, seq: u64, data: Vec<u8>) -> bool {
            if seq < self.base_seq {
                return false; // Already consumed
            }

            let offset = (seq - self.base_seq) as usize;
            if offset >= self.size {
                return false; // Too far ahead
            }

            let index = (self.head + offset) % self.size;
            if self.buffers[index].is_some() {
                return false; // Duplicate
            }

            self.buffers[index] = Some(data);
            true
        }

        fn pop(&mut self) -> Option<Vec<u8>> {
            if self.buffers[self.head].is_some() {
                let data = self.buffers[self.head].take();
                self.head = (self.head + 1) % self.size;
                self.base_seq += 1;
                data
            } else {
                None
            }
        }

        fn has_gaps(&self) -> bool {
            let mut i = self.head;
            let mut found_data = false;
            let mut found_gap_after_data = false;

            for _ in 0..self.size {
                if self.buffers[i].is_some() {
                    if found_gap_after_data {
                        return true;
                    }
                    found_data = true;
                } else if found_data {
                    found_gap_after_data = true;
                }
                i = (i + 1) % self.size;
            }

            false
        }
    }

    #[test]
    fn test_packet_buffer() {
        let mut buf = PacketBuffer::new();
        buf.copy_from(b"hello world");
        assert_eq!(buf.len(), 11);
        assert_eq!(buf.as_slice(), b"hello world");
    }

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(2, 4, 1024);

        let mut buf1 = pool.get();
        buf1.copy_from(b"test");
        assert_eq!(buf1.len(), 4);

        let _buf2 = pool.get();
        let stats = pool.stats();
        assert_eq!(stats.available, 0);
        assert_eq!(stats.allocated, 2);

        drop(buf1);
        let stats = pool.stats();
        assert_eq!(stats.available, 1);
    }

    #[test]
    fn test_ring_buffer() {
        let mut ring = PacketRingBuffer::new(4);

        // Insert in order
        assert!(ring.insert(0, vec![0]));
        assert!(ring.insert(1, vec![1]));
        assert!(ring.insert(2, vec![2]));

        // Pop in order
        assert_eq!(ring.pop(), Some(vec![0]));
        assert_eq!(ring.pop(), Some(vec![1]));
        assert_eq!(ring.pop(), Some(vec![2]));
        assert_eq!(ring.pop(), None);
    }

    #[test]
    fn test_ring_buffer_out_of_order() {
        let mut ring = PacketRingBuffer::new(4);

        // Insert out of order
        assert!(ring.insert(2, vec![2]));
        assert!(ring.insert(0, vec![0]));
        assert!(ring.insert(1, vec![1]));

        // Should still pop in order
        assert_eq!(ring.pop(), Some(vec![0]));
        assert_eq!(ring.pop(), Some(vec![1]));
        assert_eq!(ring.pop(), Some(vec![2]));
    }

    #[test]
    fn test_ring_buffer_gap_detection() {
        let mut ring = PacketRingBuffer::new(4);

        ring.insert(0, vec![0]);
        ring.insert(2, vec![2]); // Gap at 1

        assert!(ring.has_gaps());

        ring.insert(1, vec![1]);
        ring.pop(); // Remove 0

        assert!(!ring.has_gaps());
    }
}
