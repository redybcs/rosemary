use std::sync::Mutex;

use ash::{Device, vk};
use static_init::dynamic;

/// The block size. Large enough that a streaming burst is a handful of
/// allocations, small enough that a program which uses one buffer does not
/// reserve a hundred megabytes for it.
const BLOCK: u64 = 64 << 20;

/// TEMPORARY: `ROSEMARY_METAL_NOSUBALLOC=1` gives every allocation a block of
/// its own, so a fault caused by two buffers sharing memory can be told from
/// every other kind.
#[inline(always)]
fn dedicate_everything() -> bool {
    #[dynamic]
    static ON: bool = std::env::var_os("ROSEMARY_METAL_NOSUBALLOC").is_some();

    *ON
}

/// Where one buffer lives.
#[derive(Clone, Copy)]
pub struct Allocation {
    pub memory: vk::DeviceMemory,
    pub contents: *mut u8,
    /// The block this came out of, or `None` for a block of its own.
    block: Option<usize>,
    /// Which block held that index when this was carved: a spent block's
    /// index is given to the next one, and a stale free must not reach it.
    generation: u64,
    offset: u64,
    size: u64,
    /// Whether this range has never been handed out before. A driver zeroes
    /// the memory it allocates - it must, or one process would read another's -
    /// so a caller which needs zeros need only write them over a range some
    /// earlier buffer has already used.
    pub pristine: bool,
}

// SAFETY: `contents` is a permanent mapping of device memory, which every
// thread that holds the Metal object may write through - exactly as Metal's
// own `-contents` pointer is.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

struct Block {
    memory: vk::DeviceMemory,
    contents: *mut u8,
    size: u64,
    /// Free ranges, kept sorted by offset and merged with their neighbours, so
    /// a block which fills and empties does not fragment into nothing.
    free: Vec<(u64, u64)>,
    /// One past the highest offset ever carved out. Everything above it is
    /// memory as the driver handed it over, which is zeroed.
    reached: u64,
    generation: u64,
}

// SAFETY: as for `Allocation`.
unsafe impl Send for Block {}

#[derive(Default)]
struct State {
    blocks: Vec<Block>,
    /// Blocks a large allocation owns outright, by the index it was given.
    dedicated: Vec<usize>,
    generations: u64,
}

/// One memory type's blocks.
pub struct Heap {
    memory_type: u32,
    /// Whether blocks are mapped. A private heap's are not: its memory type
    /// need not be host visible, and nothing on the host touches it.
    mapped: bool,
    state: Mutex<State>,
}

impl Heap {
    pub fn new(memory_type: u32, mapped: bool) -> Self {
        Self {
            memory_type,
            mapped,
            state: Mutex::new(State::default()),
        }
    }

    /// Carve `size` bytes out, allocating another block where none has room.
    pub fn allocate(
        &self,
        device: &Device,
        size: u64,
        alignment: u64,
    ) -> Result<Allocation, String> {
        let alignment = alignment.max(1);
        let size = size.max(1).next_multiple_of(alignment);
        let mut state = self.state.lock().expect("buffer heap");
        if size <= BLOCK
            && !dedicate_everything()
            && let Some(allocation) = state.carve(size, alignment)
        {
            return Ok(allocation);
        }
        let capacity = match dedicate_everything() {
            true => size,
            false => size.max(BLOCK),
        };
        let fresh = crate::stats::timed(crate::stats::Counter::BufferBlockMicros, || {
            self.block(device, capacity)
        });
        let (block, own) = match fresh {
            Ok(block) => (block, size > BLOCK || dedicate_everything()),
            // The device would not allocate or map another block. A GPU behind
            // a virtual machine - Venus, on a Chromebook - maps only a bounded
            // window of memory, which the blocks kept mapped here fill: give
            // back every empty one, and ask for no more than this allocation.
            Err(first) if self.mapped || capacity > size => {
                for memory in state.take_empty_blocks() {
                    // SAFETY: an empty block has no live buffer bound into it,
                    // and every buffer that was is destroyed and idle.
                    unsafe { self.release_memory(device, memory) };
                }
                match self.block(device, size) {
                    Ok(block) => (block, true),
                    Err(second) => {
                        return Err(format!("{first}; and for {size} bytes alone: {second}"));
                    }
                }
            }
            Err(first) => return Err(first),
        };
        let capacity = block.size;
        let index = state.place(block);
        if own {
            // Its own block, freed when it is: nothing else is cut from it.
            state.dedicated.push(index);
            let block = &mut state.blocks[index];
            block.free.clear();
            block.reached = capacity;
            return Ok(Allocation {
                memory: block.memory,
                contents: block.contents,
                block: Some(index),
                generation: block.generation,
                offset: 0,
                size: capacity,
                pristine: true,
            });
        }
        state
            .carve(size, alignment)
            .ok_or_else(|| "a fresh block had no room for the allocation".to_owned())
    }

    /// Give a range back. The memory stays mapped and the block stays alive.
    pub fn free(&self, device: &Device, allocation: Allocation) {
        let Some(index) = allocation.block else {
            return;
        };
        let mut state = self.state.lock().expect("buffer heap");
        if !state.owns(index, allocation.generation) {
            debug_assert!(false, "a Metal heap range was freed after its block was");
            return;
        }
        if let Some(at) = state.dedicated.iter().position(|owned| *owned == index) {
            state.dedicated.swap_remove(at);
            let block = &mut state.blocks[index];
            let (memory, size) = (block.memory, block.size);
            // The block stays in place so that no later index shifts; it is
            // marked spent by having neither memory nor room.
            block.memory = vk::DeviceMemory::null();
            block.contents = std::ptr::null_mut();
            block.size = 0;
            block.free.clear();
            drop(state);
            let _ = size;
            // SAFETY: nothing else was cut from this block, and the caller
            // guarantees the buffer bound into it is destroyed and idle.
            unsafe { self.release_memory(device, memory) };
            return;
        }
        state.blocks[index].release(allocation.offset, allocation.size);
        // A burst which needed gigabytes for a second must not hold them for
        // the rest of the run. One empty block is kept, so a program which
        // fills and empties one block does not allocate on every cycle.
        let Some(spare) = state.spare_block() else {
            return;
        };
        let block = &mut state.blocks[spare];
        let memory = std::mem::replace(&mut block.memory, vk::DeviceMemory::null());
        block.contents = std::ptr::null_mut();
        block.size = 0;
        block.free.clear();
        drop(state);
        // SAFETY: the block is entirely free, so no live buffer is bound into
        // it, and every buffer that was is destroyed and idle.
        unsafe { self.release_memory(device, memory) };
    }

    /// Blocks held, their bytes, and how many of those bytes are free.
    pub fn usage(&self) -> (usize, u64, u64) {
        let state = self.state.lock().expect("buffer heap");
        let live = state.blocks.iter().filter(|block| block.size > 0);
        let free = live
            .clone()
            .map(|block| block.free.iter().map(|(_, size)| size).sum::<u64>())
            .sum();
        (
            live.clone().count(),
            live.map(|block| block.size).sum(),
            free,
        )
    }

    /// # Safety
    ///
    /// No live buffer may be bound into `memory`.
    unsafe fn release_memory(&self, device: &Device, memory: vk::DeviceMemory) {
        // SAFETY: the caller guarantees the memory is idle; it is mapped
        // exactly when the heap maps its blocks.
        unsafe {
            if self.mapped {
                device.unmap_memory(memory);
            }
            device.free_memory(memory, None);
        }
    }

    fn block(&self, device: &Device, size: u64) -> Result<Block, String> {
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(self.memory_type);
        // SAFETY: a well-formed allocation on this live device.
        let memory = unsafe { device.allocate_memory(&info, None) }
            .map_err(|error| format!("vkAllocateMemory of {size} bytes failed with {error:?}"))?;
        if !self.mapped {
            return Ok(Block {
                memory,
                contents: std::ptr::null_mut(),
                size,
                free: vec![(0, size)],
                reached: 0,
                generation: 0,
            });
        }
        // SAFETY: host-visible memory nothing else has mapped.
        let contents =
            unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(|error| {
                    // SAFETY: nothing is bound to it yet.
                    unsafe { device.free_memory(memory, None) };
                    format!("vkMapMemory failed with {error:?}")
                })? as *mut u8;
        Ok(Block {
            memory,
            contents,
            size,
            free: vec![(0, size)],
            reached: 0,
            generation: 0,
        })
    }
}

impl State {
    fn place(&mut self, mut block: Block) -> usize {
        self.generations += 1;
        block.generation = self.generations;
        match self.blocks.iter().position(|spent| spent.size == 0) {
            Some(index) => {
                self.blocks[index] = block;
                index
            }
            None => {
                self.blocks.push(block);
                self.blocks.len() - 1
            }
        }
    }

    /// Whether the block at `index` is still the one `generation` was cut from.
    fn owns(&self, index: usize, generation: u64) -> bool {
        self.blocks
            .get(index)
            .is_some_and(|block| block.size > 0 && block.generation == generation)
    }

    /// Every wholly-free shared block, taken out of service: its memory, for
    /// the caller to release.
    fn take_empty_blocks(&mut self) -> Vec<vk::DeviceMemory> {
        let mut taken = Vec::new();
        for (index, block) in self.blocks.iter_mut().enumerate() {
            let empty = block.size > 0 && block.free.len() == 1 && block.free[0] == (0, block.size);
            if !empty || self.dedicated.contains(&index) {
                continue;
            }
            taken.push(std::mem::replace(
                &mut block.memory,
                vk::DeviceMemory::null(),
            ));
            block.contents = std::ptr::null_mut();
            block.size = 0;
            block.free.clear();
        }
        taken
    }

    /// A wholly-free block to give back, once a second one is also free.
    fn spare_block(&self) -> Option<usize> {
        let mut empty = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(index, block)| {
                block.size > 0
                    && block.free.len() == 1
                    && block.free[0] == (0, block.size)
                    && !self.dedicated.contains(index)
            })
            .map(|(index, _)| index);
        let _keep = empty.next()?;
        empty.next()
    }

    /// First fit across the blocks, honouring `alignment` inside the range.
    fn carve(&mut self, size: u64, alignment: u64) -> Option<Allocation> {
        for (index, block) in self.blocks.iter_mut().enumerate() {
            if block.size == 0 {
                continue;
            }
            for (at, (start, length)) in block.free.iter().copied().enumerate() {
                let aligned = start.next_multiple_of(alignment);
                let head = aligned - start;
                if head + size > length {
                    continue;
                }
                block.free.remove(at);
                if head > 0 {
                    block.free.insert(at, (start, head));
                }
                let tail = start + length - (aligned + size);
                if tail > 0 {
                    let after = at + usize::from(head > 0);
                    block.free.insert(after, (aligned + size, tail));
                }
                let pristine = aligned >= block.reached;
                block.reached = block.reached.max(aligned + size);
                return Some(Allocation {
                    memory: block.memory,
                    contents: match block.contents.is_null() {
                        true => std::ptr::null_mut(),
                        // SAFETY: `aligned` is inside a mapping of `block.size`.
                        false => unsafe { block.contents.add(aligned as usize) },
                    },
                    block: Some(index),
                    generation: block.generation,
                    offset: aligned,
                    size,
                    pristine,
                });
            }
        }
        None
    }
}

impl Block {
    /// Return a range, merging it with whichever neighbours are also free.
    fn release(&mut self, offset: u64, size: u64) {
        let at = self.free.partition_point(|(start, _)| *start < offset);
        debug_assert!(
            (at == 0 || self.free[at - 1].0 + self.free[at - 1].1 <= offset)
                && self
                    .free
                    .get(at)
                    .is_none_or(|(next, _)| offset + size <= *next),
            "a Metal heap range was freed twice"
        );
        self.free.insert(at, (offset, size));
        // Merge forward first, so the backward merge sees the whole run.
        if at + 1 < self.free.len() {
            let (next_start, next_size) = self.free[at + 1];
            if offset + size == next_start {
                self.free[at].1 += next_size;
                self.free.remove(at + 1);
            }
        }
        if at > 0 {
            let (previous_start, previous_size) = self.free[at - 1];
            if previous_start + previous_size == offset {
                self.free[at - 1].1 += self.free[at].1;
                self.free.remove(at);
            }
        }
    }
}

impl Allocation {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The allocation of a buffer which was never given one. Freeing it is a
    /// no-op, because it holds no range of any block.
    pub fn none() -> Self {
        Self {
            memory: vk::DeviceMemory::null(),
            contents: std::ptr::null_mut(),
            block: None,
            generation: 0,
            offset: 0,
            size: 0,
            pristine: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(size: u64) -> Block {
        Block {
            memory: vk::DeviceMemory::null(),
            contents: std::ptr::null_mut(),
            size,
            free: vec![(0, size)],
            reached: 0,
            generation: 0,
        }
    }

    #[test]
    fn a_freed_range_merges_with_both_of_its_neighbours() {
        let mut block = block(300);
        block.free.clear();
        block.release(0, 100);
        block.release(200, 100);
        assert_eq!(block.free, [(0, 100), (200, 100)]);
        block.release(100, 100);
        assert_eq!(block.free, [(0, 300)]);
    }

    #[test]
    fn carving_honours_alignment_and_keeps_the_skipped_head_free() {
        let mut state = State {
            blocks: vec![block(1024)],
            dedicated: Vec::new(),
            generations: 0,
        };
        let first = state.carve(8, 1).expect("room");
        assert_eq!(first.offset, 0);
        let aligned = state.carve(16, 256).expect("room");
        assert_eq!(aligned.offset, 256);
        // The bytes between the two are still available.
        let head = state.carve(8, 1).expect("room");
        assert_eq!(head.offset, 8);
    }

    /// A range some earlier buffer used is not pristine, so its new owner
    /// zeroes it; one above everything ever carved is memory as the driver
    /// handed it over, and needs no writing at all.
    #[test]
    fn only_a_reused_range_is_reported_as_needing_zeroes() {
        let mut state = State {
            blocks: vec![block(1024)],
            dedicated: Vec::new(),
            generations: 0,
        };
        let first = state.carve(128, 1).expect("room");
        assert!(first.pristine);
        let second = state.carve(128, 1).expect("room");
        assert!(second.pristine);
        state.blocks[0].release(first.offset, first.size);
        let again = state.carve(128, 1).expect("room");
        assert_eq!(again.offset, first.offset);
        assert!(!again.pristine);
    }

    #[test]
    fn a_fresh_block_takes_the_slot_of_one_given_back() {
        let mut state = State {
            blocks: vec![block(1024), block(1024)],
            dedicated: Vec::new(),
            generations: 0,
        };
        state.blocks[0].size = 0;
        state.blocks[0].free.clear();
        assert_eq!(state.place(block(2048)), 0);
        assert_eq!(state.place(block(2048)), 2);
        assert_eq!(state.blocks.len(), 3);
    }

    #[test]
    fn a_free_from_before_its_slot_was_reused_does_not_reach_the_new_block() {
        let mut state = State::default();
        let index = state.place(block(1024));
        let stale = state.carve(128, 1).expect("room");
        state.blocks[index].size = 0;
        state.blocks[index].free.clear();
        assert_eq!(state.place(block(1024)), index);
        assert!(!state.owns(index, stale.generation));
        let live = state.carve(128, 1).expect("room");
        assert!(state.owns(index, live.generation));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "freed twice")]
    fn a_double_free_is_caught() {
        let mut state = State::default();
        state.place(block(1024));
        let taken = state.carve(128, 1).expect("room");
        state.carve(128, 1).expect("room");
        state.blocks[0].release(taken.offset, taken.size);
        state.blocks[0].release(taken.offset, taken.size);
    }

    #[test]
    fn a_block_which_fills_and_empties_is_whole_again() {
        let mut state = State {
            blocks: vec![block(1024)],
            dedicated: Vec::new(),
            generations: 0,
        };
        let mut taken = Vec::new();
        while let Some(allocation) = state.carve(128, 1) {
            taken.push(allocation);
        }
        assert_eq!(taken.len(), 8);
        for allocation in taken {
            state.blocks[0].release(allocation.offset, allocation.size);
        }
        assert_eq!(state.blocks[0].free, [(0, 1024)]);
    }
}

#[cfg(test)]
mod take_empty_tests {
    use super::*;
    use ash::vk::Handle;

    fn block(memory: u64, size: u64, free: Vec<(u64, u64)>) -> Block {
        Block {
            memory: vk::DeviceMemory::from_raw(memory),
            contents: std::ptr::null_mut(),
            size,
            free,
            reached: 0,
            generation: memory,
        }
    }

    /// When the device will not map more, only blocks nothing lives in are
    /// given back: not a block with a buffer in it, and not one a large buffer
    /// owns outright.
    #[test]
    fn take_empty_blocks_takes_only_empty_shared_blocks() {
        let mut state = State {
            blocks: vec![
                block(1, 64, vec![(0, 64)]),
                block(2, 64, vec![(16, 48)]),
                block(3, 64, vec![(0, 64)]),
                block(4, 0, vec![]),
            ],
            dedicated: vec![2],
            generations: 4,
        };
        let taken: Vec<u64> = state
            .take_empty_blocks()
            .iter()
            .map(|m| m.as_raw())
            .collect();
        assert_eq!(taken, vec![1]);
        assert_eq!(state.blocks[0].size, 0, "the empty block is spent");
        assert_eq!(
            state.blocks[1].size, 64,
            "a block with a buffer in it stays"
        );
        assert_eq!(state.blocks[2].size, 64, "a dedicated block stays");
        assert!(
            !state.owns(0, 1),
            "a range from the spent block is no longer its own"
        );
    }
}
