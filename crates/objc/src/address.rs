//! Hashing for maps keyed by an address, shared with the crates above this one.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn aligned_addresses_spread() {
        let buckets: HashSet<u64> = (0..64u64)
            .map(|i| mix(0x7f00_0000_1000 + i * 16) & 63)
            .collect();
        assert!(buckets.len() > 32, "{} of 64 buckets used", buckets.len());
    }
}
