//! Hash maps for the translator's own keys: value and block ids, types and
//! names it made itself. SipHash's flood resistance buys nothing for keys a
//! shader cannot choose, and hashing them was a sixth of translation time.

use std::hash::{BuildHasherDefault, Hasher};

pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<FastHasher>>;

/// The odd multiplier FxHash uses, whose high bits mix every input bit.
const SEED: u64 = 0xf135_7aea_2e62_a9c5;

/// Bits the final rotation brings down: the table indexes by the low bits
/// and tags by the top seven, and a product's best-mixed bits are its high ones.
const FINISH_ROTATION: u32 = 26;

#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0 ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0.rotate_left(FINISH_ROTATION)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(
                chunk.try_into().expect("an 8-byte chunk"),
            ));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word));
        }
        // Without the length, "a" and "a\0" would collide.
        self.add(bytes.len() as u64);
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_ids_fill_distinct_buckets() {
        let buckets: HashSet<u64> = (0..64u32)
            .map(|id| {
                let mut hasher = FastHasher::default();
                hasher.write_u32(id);
                hasher.finish() & 63
            })
            .collect();
        assert!(buckets.len() > 32, "{} of 64 buckets used", buckets.len());
    }

    #[test]
    fn byte_strings_differing_only_in_trailing_zeros_differ() {
        let hash = |bytes: &[u8]| {
            let mut hasher = FastHasher::default();
            hasher.write(bytes);
            hasher.finish()
        };
        assert_ne!(hash(b"a"), hash(b"a\0"));
    }
}
