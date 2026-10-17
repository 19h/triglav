//! Bounded flow affinity with idle expiry and least-recently-used eviction.
use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
struct Binding {
    uplink: u16,
    touched: Instant,
}

#[derive(Debug)]
pub(super) struct FlowCache {
    capacity: usize,
    bindings: HashMap<u64, Binding>,
    order: BTreeSet<(Instant, u64)>,
}

impl FlowCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            bindings: HashMap::new(),
            order: BTreeSet::new(),
        }
    }

    pub fn insert(&mut self, flow: u64, uplink: u16, now: Instant) -> bool {
        if self.capacity == 0 {
            return false;
        }
        self.remove(flow);
        if self.bindings.len() >= self.capacity {
            if let Some((_, evicted)) = self.order.first().copied() {
                self.remove(evicted);
            }
        }
        self.bindings.insert(
            flow,
            Binding {
                uplink,
                touched: now,
            },
        );
        self.order.insert((now, flow));
        true
    }

    pub fn get(&mut self, flow: u64, now: Instant) -> Option<u16> {
        let binding = self.bindings.get_mut(&flow)?;
        self.order.remove(&(binding.touched, flow));
        binding.touched = now;
        self.order.insert((now, flow));
        Some(binding.uplink)
    }

    pub fn remove(&mut self, flow: u64) -> Option<u16> {
        let binding = self.bindings.remove(&flow)?;
        self.order.remove(&(binding.touched, flow));
        Some(binding.uplink)
    }

    pub fn remove_uplink(&mut self, uplink: u16) {
        let flows: Vec<_> = self
            .bindings
            .iter()
            .filter(|(_, b)| b.uplink == uplink)
            .map(|(&id, _)| id)
            .collect();
        for id in flows {
            self.remove(id);
        }
    }

    pub fn expire(&mut self, now: Instant, idle: Duration) {
        while let Some((last, flow)) = self.order.first().copied() {
            if now.saturating_duration_since(last) < idle {
                break;
            }
            self.remove(flow);
        }
    }

    pub fn snapshot(&self) -> Vec<(u64, u16)> {
        self.bindings
            .iter()
            .map(|(&id, b)| (id, b.uplink))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }
    pub fn clear(&mut self) {
        self.bindings.clear();
        self.order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_flow_survives_capacity_eviction_and_idle_entries_expire() {
        let now = Instant::now();
        let mut cache = FlowCache::new(2);
        assert!(cache.insert(1, 10, now));
        assert!(cache.insert(2, 20, now + Duration::from_secs(1)));
        assert_eq!(cache.get(1, now + Duration::from_secs(2)), Some(10));
        cache.insert(3, 30, now + Duration::from_secs(3));
        assert_eq!(cache.get(2, now + Duration::from_secs(3)), None);
        assert_eq!(cache.len(), 2);
        cache.expire(now + Duration::from_secs(4), Duration::from_secs(2));
        assert_eq!(cache.snapshot(), vec![(3, 30)]);
    }

    #[test]
    fn repeated_rebind_and_removal_do_not_accumulate_recency_entries() {
        let now = Instant::now();
        let mut cache = FlowCache::new(3);
        for id in 0..10000 {
            cache.insert(id % 8, (id % 2) as u16, now + Duration::from_nanos(id));
            assert!(cache.len() <= 3);
            assert_eq!(cache.order.len(), cache.len());
        }
        cache.remove_uplink(0);
        assert!(cache.snapshot().iter().all(|(_, uplink)| *uplink == 1));
        assert_eq!(cache.order.len(), cache.len());
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.order.is_empty());
        assert!(!FlowCache::new(0).insert(1, 1, now));
    }
}
