//! Hashing for maps keyed by an address, shared with the crates above this one.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Addresses are already unique; they only need their aligned low bits spread.
#[derive(Default, Clone, Copy)]
pub struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = mix(u64::from_le_bytes(word) ^ self.0);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = mix(n);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

pub fn mix(n: u64) -> u64 {
    let product = u128::from(n) * 0x9E37_79B9_7F4A_7C15;
    (product as u64) ^ ((product >> 64) as u64)
}

pub type AddressMap<V> = HashMap<usize, V, BuildHasherDefault<AddressHasher>>;
pub type AddressSet = HashSet<usize, BuildHasherDefault<AddressHasher>>;

const SHARDS: usize = 64;

/// An [`AddressMap`] split into independently locked shards, for a side table
/// every object's deallocation consults: one lock would serialize every
/// thread freeing anything.
pub struct ShardedAddressMap<V> {
    shards: [RwLock<AddressMap<V>>; SHARDS],
}

impl<V> ShardedAddressMap<V> {
    pub const fn new() -> Self {
        Self {
            shards: [const { RwLock::new(HashMap::with_hasher(BuildHasherDefault::new())) };
                SHARDS],
        }
    }

    /// The shard holding `address`, read-locked.
    pub fn read(&self, address: usize) -> RwLockReadGuard<'_, AddressMap<V>> {
        self.shard(address)
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// The shard holding `address`, write-locked.
    pub fn write(&self, address: usize) -> RwLockWriteGuard<'_, AddressMap<V>> {
        self.shard(address)
            .write()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn shard(&self, address: usize) -> &RwLock<AddressMap<V>> {
        &self.shards[(mix(address as u64) >> 58) as usize % SHARDS]
    }
}

impl<V> Default for ShardedAddressMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A slot no address has taken, which ends a probe.
const EMPTY: usize = 0;
/// A slot whose address was removed, which a probe passes over. Nothing kept
/// here is this small: the addresses are of word-aligned structures.
const REMOVED: usize = 1;
/// The fewest slots a table has.
const MIN_SLOTS: usize = 256;

/// A set of addresses whose lookups take no lock: one asked about every
/// object retain and release see, and changed a handful of times per image.
///
/// Open addressing, probed linearly, kept at most half full so that every
/// probe ends at an empty slot. A table that fills is copied into a larger
/// one and leaked rather than freed, because a reader may still be probing
/// it; the tables together are under twice the size of the last.
pub struct LockFreeAddressSet {
    table: AtomicPtr<Table>,
    /// Held by a writer, with the number of slots the current table has used.
    used: Mutex<usize>,
}

struct Table {
    /// A power of two of them.
    slots: Box<[AtomicUsize]>,
}

impl Table {
    fn with_slots(count: usize) -> Self {
        Self {
            slots: (0..count).map(|_| AtomicUsize::new(EMPTY)).collect(),
        }
    }

    /// The slot holding `address`, or the empty one ending its probe.
    fn find(&self, address: usize, ordering: Ordering) -> (usize, usize) {
        let mask = self.slots.len() - 1;
        let mut index = mix(address as u64) as usize & mask;
        loop {
            match self.slots[index].load(ordering) {
                found @ EMPTY => return (index, found),
                found if found == address => return (index, found),
                _ => index = (index + 1) & mask,
            }
        }
    }
}

impl LockFreeAddressSet {
    pub const fn new() -> Self {
        Self {
            table: AtomicPtr::new(std::ptr::null_mut()),
            used: Mutex::new(0),
        }
    }

    pub fn contains(&self, address: usize) -> bool {
        let table = self.table.load(Ordering::Acquire);
        // SAFETY: a published table is never freed.
        address > REMOVED
            && !table.is_null()
            && unsafe { &*table }.find(address, Ordering::Acquire).1 == address
    }

    pub fn insert(&self, address: usize) {
        debug_assert!(address > REMOVED, "{address:#x} is a reserved slot value");
        let mut used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        let mut table = self.table.load(Ordering::Relaxed);
        // SAFETY: as in `contains`; and only the lock holder replaces it.
        if table.is_null() || (*used + 1) * 2 > unsafe { &*table }.slots.len() {
            table = self.grow(table, &mut used);
        }
        // SAFETY: as above.
        let table = unsafe { &*table };
        let (index, found) = table.find(address, Ordering::Relaxed);
        if found == EMPTY {
            // Release: a reader that finds the address sees what was written
            // at it before it was inserted.
            table.slots[index].store(address, Ordering::Release);
            *used += 1;
        }
    }

    pub fn remove(&self, address: usize) {
        let _used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        let table = self.table.load(Ordering::Relaxed);
        if table.is_null() {
            return;
        }
        // SAFETY: as in `contains`.
        let table = unsafe { &*table };
        let (index, found) = table.find(address, Ordering::Relaxed);
        if found == address {
            table.slots[index].store(REMOVED, Ordering::Release);
        }
    }

    /// Publish a table holding `old`'s addresses with room for as many again,
    /// and count its slots used.
    fn grow(&self, old: *mut Table, used: &mut usize) -> *mut Table {
        // SAFETY: as in `contains`.
        let live: Vec<usize> = match unsafe { old.as_ref() } {
            Some(old) => old
                .slots
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .filter(|&address| address != EMPTY && address != REMOVED)
                .collect(),
            None => Vec::new(),
        };
        let table = Table::with_slots(((live.len() + 1) * 4).next_power_of_two().max(MIN_SLOTS));
        for &address in &live {
            let (index, _) = table.find(address, Ordering::Relaxed);
            table.slots[index].store(address, Ordering::Relaxed);
        }
        *used = live.len();
        let table = Box::into_raw(Box::new(table));
        // Release: a reader that loads the table sees every slot written above.
        self.table.store(table, Ordering::Release);
        table
    }
}

impl Default for LockFreeAddressSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn a_lock_free_set_keeps_what_was_inserted_through_growth_and_removal() {
        let set = LockFreeAddressSet::new();
        assert!(!set.contains(0x1000));
        let addresses: Vec<usize> = (1..5000).map(|i| 0x7f00_0000_0000 + i * 64).collect();
        for &address in &addresses {
            set.insert(address);
        }
        set.insert(addresses[0]);
        assert!(addresses.iter().all(|&address| set.contains(address)));
        assert!(!set.contains(0x7f00_0000_0000 + 8));
        for &address in addresses.iter().step_by(2) {
            set.remove(address);
        }
        for (index, &address) in addresses.iter().enumerate() {
            assert_eq!(set.contains(address), index % 2 == 1, "{address:#x}");
        }
        set.insert(addresses[0]);
        assert!(set.contains(addresses[0]));
    }

    #[test]
    fn a_lock_free_set_answers_readers_while_it_grows() {
        let set = LockFreeAddressSet::new();
        let base = 0x5500_0000_0000usize;
        set.insert(base);
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    while !done.load(Ordering::Relaxed) {
                        assert!(set.contains(base));
                        assert!(!set.contains(base + 8));
                    }
                });
            }
            for i in 1..20_000 {
                set.insert(base + i * 16);
            }
            done.store(true, Ordering::Relaxed);
        });
        assert!((0..20_000).all(|i| set.contains(base + i * 16)));
    }

    /// What asking a side table about an address costs, locked and not, with
    /// every core asking at once. Run with `cargo test --release -p objc --
    /// --ignored --nocapture side_table_lookup_cost`.
    #[test]
    #[ignore = "a benchmark, not a check"]
    fn side_table_lookup_cost() {
        const LOOKUPS: usize = 5_000_000;
        let classes: Vec<usize> = (1..2000).map(|i| 0x7f00_0000_0000 + i * 4096).collect();
        let sharded = ShardedAddressMap::<()>::new();
        let lock_free = LockFreeAddressSet::new();
        for &class in &classes {
            sharded.write(class).insert(class, ());
            lock_free.insert(class);
        }
        let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
        let time = |name: &str, contains: &(dyn Fn(usize) -> bool + Sync)| {
            let start = std::time::Instant::now();
            std::thread::scope(|scope| {
                for thread in 0..threads {
                    scope.spawn(move || {
                        let mut found = 0;
                        for i in 0..LOOKUPS {
                            // Mostly objects, which are not classes, as retain sees.
                            let address = 0x7f00_0000_0010 + (i * 48 + thread * 8) % (1 << 26);
                            found += usize::from(contains(std::hint::black_box(address)));
                        }
                        std::hint::black_box(found);
                    });
                }
            });
            eprintln!(
                "{name}: {:.2} ns per lookup, {threads} threads",
                start.elapsed().as_nanos() as f64 / LOOKUPS as f64
            );
        };
        time("sharded rwlock", &|address| {
            sharded.read(address).contains_key(&address)
        });
        time("lock-free", &|address| lock_free.contains(address));
    }

    #[test]
    fn aligned_addresses_spread() {
        let buckets: HashSet<u64> = (0..64u64)
            .map(|i| mix(0x7f00_0000_1000 + i * 16) & 63)
            .collect();
        assert!(buckets.len() > 32, "{} of 64 buckets used", buckets.len());
    }
}
