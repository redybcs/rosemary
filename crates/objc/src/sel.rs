use std::collections::HashSet;
use std::ffi::{CStr, c_char};
use std::hash::BuildHasherDefault;
use std::sync::RwLock;
use std::sync::atomic::{AtomicPtr, Ordering};

pub type Sel = *const c_char;

/// Every name registered so far, by its bytes, leaked with a NUL after them so
/// that a name's address is its selector forever. Keyed by the bytes so that a
/// lookup needs no NUL-terminated copy of a Rust name, and hashed as the
/// address maps are: the names are the program's own.
static TABLE: RwLock<HashSet<&'static [u8], BuildHasherDefault<crate::address::AddressHasher>>> =
    RwLock::new(HashSet::with_hasher(BuildHasherDefault::new()));

/// The selector for `name`, registering it if this is the first sighting.
pub fn register(name: &CStr) -> Sel {
    intern(name.to_bytes())
}

/// The selector named `bytes`, which hold no NUL.
fn intern(bytes: &[u8]) -> Sel {
    if let Some(found) = TABLE.read().unwrap_or_else(|e| e.into_inner()).get(bytes) {
        return found.as_ptr().cast();
    }
    let mut table = TABLE.write().unwrap_or_else(|e| e.into_inner());
    // Re-checked under the write lock: two threads can miss the read.
    if let Some(found) = table.get(bytes) {
        return found.as_ptr().cast();
    }
    let terminated: &'static [u8] = Box::leak([bytes, &[0]].concat().into_boxed_slice());
    let name = &terminated[..bytes.len()];
    table.insert(name);
    name.as_ptr().cast()
}

const RECENT_SLOTS: usize = 256;

thread_local! {
    static RECENT: [std::cell::Cell<(usize, usize, Sel)>; RECENT_SLOTS] =
        const { [const { std::cell::Cell::new((0, 0, std::ptr::null())) }; RECENT_SLOTS] };
}

/// The selector for a Rust string, for this crate's own classes.
///
/// Panics on an interior NUL, which only a bug here could produce.
pub fn get(name: &str) -> Sel {
    let address = name.as_ptr() as usize;
    let slot = (crate::address::mix((address ^ name.len()) as u64) >> 56) as usize % RECENT_SLOTS;
    let (at, length, sel) = RECENT.with(|recent| recent[slot].get());
    // The bytes are compared because the buffer at `address` may have been
    // reused for another name since.
    // SAFETY: a non-null cached selector was interned from a name of exactly
    // `length` bytes, so that many are readable before its NUL.
    if at == address
        && length == name.len()
        && !sel.is_null()
        && unsafe { std::slice::from_raw_parts(sel.cast::<u8>(), length) } == name.as_bytes()
    {
        return sel;
    }
    assert!(
        !name.as_bytes().contains(&0),
        "a selector name has no interior NUL"
    );
    let sel = intern(name.as_bytes());
    RECENT.with(|recent| recent[slot].set((address, name.len(), sel)));
    sel
}

/// [`register`] for [`sel!`], which remembers the answer in `cache`.
#[doc(hidden)]
pub fn register_cached(cache: &AtomicPtr<c_char>, name: &'static str) -> Sel {
    let sel = get(name);
    cache.store(sel.cast_mut(), Ordering::Release);
    sel
}

/// The selector for a string literal, registered on the first send from each
/// call site and loaded from a static after that: the constant selectors on
/// hot paths need neither a hash nor a comparison of their name.
#[macro_export]
macro_rules! sel {
    ($name:literal) => {{
        static CACHE: ::std::sync::atomic::AtomicPtr<::std::ffi::c_char> =
            ::std::sync::atomic::AtomicPtr::new(::std::ptr::null_mut());
        let sel = CACHE.load(::std::sync::atomic::Ordering::Acquire);
        if sel.is_null() {
            $crate::sel::register_cached(&CACHE, $name)
        } else {
            sel.cast_const()
        }
    }};
}

/// A selector's name, which is the selector.
pub fn name(sel: Sel) -> *const c_char {
    validate(sel);
    sel
}

pub(crate) fn validate(sel: Sel) {
    if crate::debug::enabled()
        && !sel.is_null()
        && !TABLE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|known| known.as_ptr().cast() == sel)
    {
        crate::fail!("objc debug: invalid selector handle {sel:p}");
    }
}

/// Whether two selectors are the same.
///
/// Pointer equality, because registration is what makes them equal. Null is
/// compared as a value so `sel_isEqual(NULL, NULL)` is true, as it is on Apple's.
pub fn is_equal(a: Sel, b: Sel) -> bool {
    validate(a);
    validate(b);
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reused_name_buffer_gets_its_new_selector() {
        let mut name = String::with_capacity(32);
        name.push_str("cacheProbeAlpha");
        let alpha = get(&name);
        name.clear();
        name.push_str("cacheProbeBravo");
        let bravo = get(&name);
        assert_ne!(alpha, bravo);
        assert_eq!(bravo, register(c"cacheProbeBravo"));
        assert_eq!(alpha, register(c"cacheProbeAlpha"));
    }

    #[test]
    fn a_literal_selector_is_the_registered_one() {
        let sel = crate::sel!("cacheProbeLiteral");
        assert_eq!(sel, get("cacheProbeLiteral"));
        assert_eq!(sel, crate::sel!("cacheProbeLiteral"));
    }

    #[test]
    fn a_name_registered_from_c_and_from_rust_is_one_selector() {
        let from_c = register(c"cacheProbeBothWays:");
        assert_eq!(get("cacheProbeBothWays:"), from_c);
        // SAFETY: a selector is a NUL-terminated interned name.
        assert_eq!(unsafe { CStr::from_ptr(from_c) }, c"cacheProbeBothWays:");
        let empty = get("");
        assert_eq!(empty, register(c""));
        // SAFETY: as above.
        assert_eq!(unsafe { CStr::from_ptr(empty) }, c"");
    }

    #[test]
    fn the_same_name_registers_to_the_same_pointer() {
        let a = get("countByEnumeratingWithState:objects:count:");
        let b = get("countByEnumeratingWithState:objects:count:");
        assert_eq!(a, b);
        assert_ne!(a, get("count"));
    }

    #[test]
    fn a_selector_is_readable_as_its_own_name() {
        let sel = get("initWithBytes:length:encoding:");
        // SAFETY: `register` leaked a NUL-terminated copy of the name.
        let read = unsafe { CStr::from_ptr(name(sel)) };
        assert_eq!(read.to_str().unwrap(), "initWithBytes:length:encoding:");
    }
}
