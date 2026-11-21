//! CF objects whose contents are a boxed Rust value.
//!
//! Shared so that every such type gives its contents back the same way: one
//! `-dealloc`, written once, rather than a copy per framework crate.

use std::ffi::c_void;

use objc::class::Class;
use objc::object::Id;

use crate::base::CFTypeRef;

/// The object, and a pointer to the Rust value holding its contents.
#[repr(C)]
struct Body<T> {
    isa: *const Class,
    contents: *mut T,
}

/// Register a class whose instances hold a `T`, with a `-dealloc` that gives
/// the contents back. `is_type` is [`crate::base::register`]'s.
///
/// Each instance keeps its reference count in a word after the body, as
/// Metal's objects do: arrays, dictionaries and the like are retained and
/// released constantly, and the runtime's shared table took a lock for each.
/// Only [`make`] (or the runtime's own allocation, zeroed) makes an instance,
/// and no literal in an image has one of these classes, which is what
/// [`Class::count_inline`] requires. A subclass, a guest's included, keeps
/// counting in the table.
pub fn register<T: 'static>(
    name: &'static std::ffi::CStr,
    superclass: Option<&'static Class>,
    is_type: bool,
) -> &'static Class {
    let fresh = objc::class::lookup(name).is_none();
    let word = size_of::<usize>();
    let size = size_of::<Body<T>>().next_multiple_of(word) + word;
    let class = crate::base::register(name, superclass, size, is_type);
    if fresh && class.instance_size() == size {
        class.count_inline();
    }
    unsafe extern "C" fn dealloc<T>(receiver: Id, _sel: objc::sel::Sel) {
        // SAFETY: `-dealloc` is reached with the last reference gone, and the
        // receiver is a `Body<T>` whose `contents` came from `Box::into_raw`.
        unsafe {
            let body = receiver as *mut Body<T>;
            if !(*body).contents.is_null() {
                drop(Box::from_raw((*body).contents));
                (*body).contents = std::ptr::null_mut();
            }
            objc::arc::finalize(receiver);
        }
    }
    class.replace_method(
        objc::sel::get("dealloc"),
        dealloc::<T> as *const c_void,
        c"v16@0:8",
    );
    class
}

/// Allocate an instance of `class` holding `contents`, or null.
pub fn make<T>(class: &'static Class, contents: T) -> CFTypeRef {
    // SAFETY: `Body<T>` begins with `isa` and is the layout `class` was
    // registered with.
    let body: *mut Body<T> = unsafe { crate::base::allocate(class) };
    if body.is_null() {
        return std::ptr::null();
    }
    // SAFETY: a live instance whose second word is ours to fill in.
    unsafe { (*body).contents = Box::into_raw(Box::new(contents)) };
    body as CFTypeRef
}

/// The contents of `object`, if it is an instance of `class`.
///
/// # Safety
///
/// `object` must be null or an object this runtime allocated, and `class` one
/// [`register`] made for `T`.
pub unsafe fn contents<'a, T>(object: CFTypeRef, class: &'static Class) -> Option<&'a mut T> {
    if !unsafe { crate::base::is_kind_of(object, class) } {
        return None;
    }
    // SAFETY: an instance of `class` is a `Body<T>` with a live `contents`.
    unsafe {
        let body = object as *mut Body<T>;
        (!(*body).contents.is_null()).then(|| &mut *(*body).contents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DROPPED: AtomicUsize = AtomicUsize::new(0);

    struct Counted;

    impl Drop for Counted {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A boxed object counts in its own word: retains and releases balance
    /// there, and its contents go exactly at the last release.
    #[test]
    fn a_boxed_object_counts_inline_and_frees_at_the_last_release() {
        let class = register::<Counted>(c"RosemaryBoxedInlineTest", None, false);
        assert!(
            class.inline_count_offset().is_some(),
            "a boxed class counts inline"
        );
        let object = make(class, Counted) as Id;
        assert_eq!(objc::arc::retain_count(object), 1);
        objc::arc::retain(object);
        objc::arc::retain(object);
        assert_eq!(objc::arc::retain_count(object), 3);
        objc::arc::release(object);
        objc::arc::release(object);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            0,
            "freed before the last release"
        );
        objc::arc::release(object);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            1,
            "contents not freed at the last release"
        );
    }
}
