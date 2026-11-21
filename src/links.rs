//! Exact linked-id resolution, built in four parallel passes:
//! 1. scatter endpoint occurrences into partitions by top hash bits;
//! 2. intern each partition, turning occurrences into dense vertex indices;
//! 3. replay the edges through those indices into a lock-free disjoint set
//!    whose roots are always their component's bytewise-smallest id;
//! 4. write every id named by another id into one flat table in hash order.

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Barrier, Mutex};
use std::thread;

use super::stream::hash;
use super::{SlowIntern, cmp_keys};

/// One renamed id as `[hash, name]`. The hash is `stream::hash(key)`, a
/// bijection with hash(0) == 0: it identifies the key, and zero marks a hole.
/// A plain array, so a table of holes comes straight from zeroed pages.
type Slot = [u64; 2];

/// Slots a lookup compares at once, without branching.
const WINDOW: usize = 4;

/// Immutable id -> name lookup. Only ids named by another id are stored:
/// absent ids and component minima both resolve to themselves.
pub(super) struct Aliases {
    // Robin Hood layout: hashes descend along the table and each key sits at
    // or after its home. Homes rise with the top hash bits, so one user bin
    // reads one contiguous stretch, and a window whose last hash is not
    // above the query's (holes are zero) decides the lookup.
    table: Box<[Slot]>,
    cap: u64,
}

#[inline(always)]
fn home(h: u64, cap: u64) -> usize {
    ((u128::from(!h) * u128::from(cap)) >> 64) as usize
}

impl Aliases {
    /// Return the bytewise-smallest ID in `k`'s linked component. IDs absent
    /// from all link endpoints are singleton components and resolve to self.
    #[inline(always)]
    pub(super) fn resolve(&self, k: u64) -> u64 {
        debug_assert_ne!(k, 0);
        let h = hash(k);
        let at = home(h, self.cap);
        // SAFETY: `at < cap` (or 0 when cap is 0), and the table always holds
        // at least `cap + WINDOW` slots.
        let w = unsafe { &*self.table.as_ptr().add(at).cast::<[Slot; WINDOW]>() };
        let mut name = k;
        for s in w {
            // About a third of lookups hit, at no predictable slot: a branch
            // here would mispredict constantly.
            name = std::hint::select_unpredictable(s[0] == h, s[1], name);
        }
        if w[WINDOW - 1][0] > h {
            return self.far(h, at + WINDOW, k);
        }
        name
    }

    // Long clusters only: continue the scan past the window.
    #[cold]
    #[inline(never)]
    fn far(&self, h: u64, mut at: usize, k: u64) -> u64 {
        // The table ends with WINDOW holes, so the scan stops in bounds.
        loop {
            let [hash, name] = self.table[at];
            if hash == h {
                return name;
            }
            if hash < h {
                return k;
            }
            at += 1;
        }
    }
}

/// Robin Hood insertion into `slots`, which starts at table index `offset`.
/// Returns the element that would leave the slice: the caller places it
/// later, when the following slice is no longer being written.
#[inline]
fn place(slots: &mut [Slot], offset: usize, cap: u64, mut e: Slot) -> Option<Slot> {
    let mut at = home(e[0], cap) - offset;
    while at < slots.len() {
        let s = slots[at];
        if s[0] < e[0] {
            slots[at] = e;
            if s[0] == 0 {
                return None;
            }
            e = s;
        }
        at += 1;
    }
    Some(e)
}

#[inline(always)]
fn part_of(h: u64, parts: usize) -> usize {
    ((u128::from(h) * parts as u128) >> 64) as usize
}

/// Hint that `x` is needed soon. Prefetches never fault and have no
/// observable effect.
#[inline(always)]
fn prefetch<T>(x: &T) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM only hints the cache; `x` is a valid reference.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) x as *const T, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 only hints the cache; `x` is a valid reference.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>((x as *const T).cast());
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = x;
}

/// A disjoint-set node holds its parent's index while linked, or its own key
/// while a root. Keys are never below 2^32 (packed keys carry a nonzero
/// prefix rank in bits 60..63, interned ones bit 63), so the two never mix,
/// and a root's name arrives with the load that finds it.
#[inline(always)]
fn is_key(v: u64) -> bool {
    v > u64::from(u32::MAX)
}

/// Find the root and its key. Nearly every vertex is within two links of
/// its root, which this resolves without a data-dependent branch (a root's
/// own node is simply reloaded).
#[inline(always)]
fn root(node: &[AtomicU64], x: u32) -> (u32, u64) {
    let v = node[x as usize].load(Relaxed);
    let p = std::hint::select_unpredictable(is_key(v), x, v as u32);
    let w = node[p as usize].load(Relaxed);
    let q = std::hint::select_unpredictable(is_key(w), p, w as u32);
    let u = node[q as usize].load(Relaxed);
    if is_key(u) {
        return (q, u);
    }
    deep(node, q)
}

/// Longer paths, with path halving. Parents only ever move to smaller keys,
/// so a stale relaxed load still points at an ancestor.
#[cold]
#[inline(never)]
fn deep(node: &[AtomicU64], mut x: u32) -> (u32, u64) {
    loop {
        let p = node[x as usize].load(Relaxed);
        if is_key(p) {
            return (x, p);
        }
        let g = node[p as usize].load(Relaxed);
        if is_key(g) {
            return (p as u32, g);
        }
        node[x as usize].store(g, Relaxed);
        x = g as u32;
    }
}

/// Lock-free union. The root with the larger key becomes a child of the
/// other, so every parent has a smaller key than its child: concurrent links
/// cannot form a cycle, and each root is its component's minimum. Returns
/// whether this call linked two roots. Without other workers (`SHARED` off)
/// the root found is still a root, so a plain store links it.
#[inline(always)]
fn unite<const SHARED: bool>(node: &[AtomicU64], slow: &SlowIntern, a: u32, b: u32) -> bool {
    let (mut ra, mut ka) = root(node, a);
    let (mut rb, mut kb) = root(node, b);
    loop {
        if ra == rb {
            return false;
        }
        let a_first = cmp_keys(ka, kb, slow) == Ordering::Less;
        let (lo, hi, key) = std::hint::select_unpredictable(a_first, (ra, rb, kb), (rb, ra, ka));
        if !SHARED {
            node[hi as usize].store(u64::from(lo), Relaxed);
            return true;
        }
        if node[hi as usize].compare_exchange(key, u64::from(lo), Relaxed, Relaxed).is_ok() {
            return true;
        }
        (ra, ka) = root(node, ra);
        (rb, kb) = root(node, rb);
    }
}

/// Replay one chunk of edges into the disjoint set, starting each
/// partition's cursor at the chunk's offset into the scattering worker's
/// index bins. Returns how many roots it linked.
fn join_chunk<const SHARED: bool>(
    chunk: &[(u64, u64)],
    ids: &[Vec<u32>],
    cursor: &mut [usize],
    base: &[usize],
    node: &[AtomicU64],
    slow: &SlowIntern,
) -> usize {
    let np = ids.len();
    let mut next = |k: u64| {
        let p = part_of(hash(k), np);
        let local = ids[p][cursor[p]];
        cursor[p] += 1;
        (base[p] + local as usize) as u32
    };
    // A short software pipeline: resolve ids and prefetch both nodes LAG
    // edges before they are joined, and their parents LAG / 2 edges before.
    const LAG: usize = 32;
    let mut ring = [(0u32, 0u32); LAG];
    let up = |x: u32| {
        let v = node[x as usize].load(Relaxed);
        prefetch(&node[std::hint::select_unpredictable(is_key(v), x as usize, v as usize)]);
    };
    let mut linked = 0;
    for (n, &(a, b)) in chunk.iter().enumerate() {
        let (ga, gb) = (next(a), next(b));
        prefetch(&node[ga as usize]);
        prefetch(&node[gb as usize]);
        if n >= LAG / 2 {
            let (ha, hb) = ring[(n - LAG / 2) % LAG];
            up(ha);
            up(hb);
        }
        let slot = &mut ring[n % LAG];
        if n >= LAG {
            linked += usize::from(unite::<SHARED>(node, slow, slot.0, slot.1));
        }
        *slot = (ga, gb);
    }
    for i in chunk.len().saturating_sub(LAG)..chunk.len() {
        let (ga, gb) = ring[i % LAG];
        linked += usize::from(unite::<SHARED>(node, slow, ga, gb));
    }
    linked
}

/// Intern one partition's endpoint occurrences: each key becomes its
/// partition-local index, bin by bin. Returns the distinct keys in index
/// order and the index bins.
fn intern(bins: Vec<Vec<u64>>, seen: &mut Vec<u64>, index: &mut Vec<u32>) -> (Vec<u64>, Vec<Vec<u32>>) {
    let occurrences = bins.iter().map(Vec::len).sum::<usize>();
    let cap = (occurrences * 4).next_power_of_two().max(16);
    if seen.len() < cap {
        seen.resize(cap, 0);
        index.resize(cap, 0);
    }
    let mask = cap - 1;
    let shift = 64 - cap.trailing_zeros();
    let mut keys = vec![0u64; occurrences];
    let mut n = 0;
    let mut ids = Vec::with_capacity(bins.len());
    for bin in bins {
        let mut out = Vec::with_capacity(bin.len());
        for &k in &bin {
            // Keys' middle bits are already random: one multiply spreads
            // them over the slot bits.
            let mut at = (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> shift) as usize;
            while seen[at] != k && seen[at] != 0 {
                at = (at + 1) & mask;
            }
            // New and repeated keys are equally likely: no branch on which.
            let new = seen[at] == 0;
            let id = std::hint::select_unpredictable(new, n as u32, index[at]);
            seen[at] = k;
            index[at] = id;
            keys[n] = k;
            n += usize::from(new);
            out.push(id);
        }
        ids.push(out);
    }
    seen[..cap].fill(0);
    keys.truncate(n);
    assert!(keys.iter().all(|&k| is_key(k)), "id key collides with a vertex index");
    (keys, ids)
}

/// Hand out jobs to `workers` threads; each job is taken exactly once.
fn each<J: Send, S: Send>(jobs: Vec<J>, workers: usize, init: impl Fn() -> S + Sync, run: impl Fn(&mut S, J) + Sync) -> Vec<S> {
    let queue = Mutex::new(jobs.into_iter());
    thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut s = init();
                    loop {
                        let job = queue.lock().unwrap().next();
                        match job {
                            Some(j) => run(&mut s, j),
                            None => break,
                        }
                    }
                    s
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Build an exact resolver from link endpoints only. The table is one array
/// ordered by hash, so any user bin (a range of top hash bits) maps to one
/// contiguous run of it, whatever `_user_bins` is.
pub(super) fn build(
    parts: Vec<Vec<(u64, u64)>>,
    slow: &SlowIntern,
    threads: usize,
    _user_bins: usize,
) -> Aliases {
    let edge_count = parts.iter().map(Vec::len).sum::<usize>();
    if edge_count == 0 {
        return Aliases { table: vec![[0; 2]; WINDOW].into_boxed_slice(), cap: 0 };
    }
    // Thread start-up costs more than a few thousand edges.
    let workers = threads.clamp(1, edge_count.div_ceil(1 << 13));
    let np = (edge_count / 2048).next_power_of_two().clamp(1, 512);

    // Edges go out in small chunks, so a slow or preempted worker cannot
    // stall a pass. A worker scatters each chunk it takes into its own
    // partition bins and notes where the chunk starts in each bin; the union
    // pass replays a chunk from those offsets, on whichever worker.
    const CHUNK: usize = 4096;
    let chunks: Vec<&[(u64, u64)]> = parts.iter().flat_map(|p| p.chunks(CHUNK)).collect();
    let hint = 2 * edge_count.div_ceil(workers) / np + 16;
    let next = AtomicUsize::new(0);
    // Per worker: its bins, and (chunk, offset into each bin) per chunk taken.
    type Scan = (Vec<Vec<u64>>, Vec<(usize, Vec<u32>)>);
    let scans: Vec<Scan> = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut bins: Vec<Vec<u64>> = (0..np).map(|_| Vec::with_capacity(hint)).collect();
                    let mut starts = Vec::new();
                    loop {
                        let c = next.fetch_add(1, Relaxed);
                        let Some(&chunk) = chunks.get(c) else { break };
                        starts.push((c, bins.iter().map(|b| b.len() as u32).collect()));
                        for &(a, b) in chunk {
                            debug_assert!(a != 0 && b != 0);
                            bins[part_of(hash(a), np)].push(a);
                            bins[part_of(hash(b), np)].push(b);
                        }
                    }
                    (bins, starts)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut scattered = Vec::with_capacity(workers);
    let mut origin: Vec<(usize, Vec<u32>)> = (0..chunks.len()).map(|_| (0, Vec::new())).collect();
    for (w, (bins, starts)) in scans.into_iter().enumerate() {
        scattered.push(bins);
        for (c, at) in starts {
            origin[c] = (w, at);
        }
    }

    // Intern each partition with one reusable scratch table per worker.
    let jobs: Vec<(usize, Vec<Vec<u64>>)> = (0..np)
        .map(|p| (p, scattered.iter_mut().map(|bins| std::mem::take(&mut bins[p])).collect()))
        .collect();
    let done = each(jobs, workers.min(np), || (Vec::new(), Vec::new(), Vec::new()), |(seen, index, out), (p, bins)| {
        let (keys, ids) = intern(bins, seen, index);
        out.push((p, keys, ids));
    });
    drop(scattered);
    let mut uniq: Vec<Vec<u64>> = (0..np).map(|_| Vec::new()).collect();
    let mut scattered: Vec<Vec<Vec<u32>>> = (0..workers).map(|_| (0..np).map(|_| Vec::new()).collect()).collect();
    for (_, _, out) in done {
        for (p, keys, ids) in out {
            uniq[p] = keys;
            for (w, bin) in ids.into_iter().enumerate() {
                scattered[w][p] = bin;
            }
        }
    }
    let mut base = Vec::with_capacity(np + 1);
    base.push(0usize);
    for keys in &uniq {
        base.push(base.last().unwrap() + keys.len());
    }
    let vertices = base[np];
    assert!(vertices < u32::MAX as usize, "too many linked aliases");

    // Every vertex starts as a root holding its key; each worker copies a
    // share of them. Then chunks are replayed: cursors starting at a chunk's
    // offsets meet the interned indices in the order they were scattered, so
    // no endpoint is looked up again.
    let mut cells = vec![0u64; vertices];
    const _: () = assert!(std::mem::align_of::<AtomicU64>() == std::mem::align_of::<u64>());
    // SAFETY: AtomicU64 has the size and (asserted) alignment of u64, and
    // `cells` is only accessed through this view until the view is dead.
    let node: &[AtomicU64] = unsafe { std::slice::from_raw_parts(cells.as_mut_ptr().cast(), vertices) };
    let renamed = AtomicUsize::new(0);
    let ready = Barrier::new(workers);
    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        for w in 0..workers {
            let (base, uniq, scattered, origin, chunks) = (&base, &uniq, &scattered, &origin, &chunks);
            let (renamed, ready, next) = (&renamed, &ready, &next);
            scope.spawn(move || {
                for p in w * np / workers..(w + 1) * np / workers {
                    for (cell, &k) in node[base[p]..base[p + 1]].iter().zip(&uniq[p]) {
                        cell.store(k, Relaxed);
                    }
                }
                ready.wait();
                let mut cursor = vec![0usize; np];
                let mut linked = 0;
                loop {
                    let c = next.fetch_add(1, Relaxed);
                    let Some(&chunk) = chunks.get(c) else { break };
                    let (owner, at) = &origin[c];
                    for (cur, &a) in cursor.iter_mut().zip(at) {
                        *cur = a as usize;
                    }
                    let ids = &scattered[*owner];
                    linked += if workers == 1 {
                        join_chunk::<false>(chunk, ids, &mut cursor, base, node, slow)
                    } else {
                        join_chunk::<true>(chunk, ids, &mut cursor, base, node, slow)
                    };
                }
                renamed.fetch_add(linked, Relaxed);
            });
        }
    });
    drop(scattered);
    let renamed = renamed.into_inner();

    // Half-full table; a multiple of `np` so partitions own whole slot runs.
    let cap = (renamed * 2).div_ceil(np).max(1) * np;
    let width = cap / np;
    // Spare holes at the end absorb clusters spilling past `cap`, so the
    // table practically never has to grow (and be copied).
    let mut table = vec![[0u64; 2]; cap + WINDOW + 64];
    // Partition p holds the hashes whose complement lies in run np - 1 - p.
    let runs: Vec<(usize, &mut [Slot])> = table[..cap].chunks_mut(width).enumerate().map(|(q, s)| (np - 1 - q, s)).collect();
    let spilled = each(runs, workers.min(np), || (Vec::new(), Vec::new()), |(spill, found), (p, slots)| {
        let offset = (np - 1 - p) * width;
        let keys = &uniq[p];
        // Gather renamed ids without branching on which they are, in three
        // stages LAG vertices apart: step to the parent and fetch it, step
        // on to the grandparent and fetch it, then read the root. Finds of
        // depth two thus rarely wait on memory.
        const LAG: usize = 32;
        const RING: usize = 2 * LAG;
        let (first, len) = (base[p], keys.len());
        let up = |x: usize| {
            let v = node[x].load(Relaxed);
            std::hint::select_unpredictable(is_key(v), x, v as usize)
        };
        let (mut parent, mut grand) = ([0usize; RING], [0usize; RING]);
        for i in 0..RING.min(len) {
            parent[i] = up(first + i);
            prefetch(&node[parent[i]]);
        }
        for i in 0..LAG.min(len) {
            grand[i] = up(parent[i]);
            prefetch(&node[grand[i]]);
        }
        found.resize(len, [0; 2]);
        let mut n = 0;
        for (i, &k) in keys.iter().enumerate() {
            if i + LAG < len {
                let j = (i + LAG) % RING;
                grand[j] = up(parent[j]);
                prefetch(&node[grand[j]]);
            }
            let j = i % RING;
            let q = grand[j];
            if i + RING < len {
                parent[j] = up(first + i + RING);
                prefetch(&node[parent[j]]);
            }
            let u = node[q].load(Relaxed);
            let (r, name) = if is_key(u) { (q as u32, u) } else { deep(node, q as u32) };
            found[n] = [hash(k), name];
            n += usize::from(r as usize != first + i);
        }
        for &e in &found[..n] {
            if let Some(e) = place(slots, offset, cap as u64, e) {
                spill.push(e);
            }
        }
    });
    drop(cells);
    for (spill, _) in spilled {
        for e in spill {
            settle(&mut table, cap as u64, e);
        }
    }
    Aliases { table: table.into_boxed_slice(), cap: cap as u64 }
}

/// Robin Hood insertion anywhere in the table, growing it so that at least
/// WINDOW holes always follow the last occupied slot.
fn settle(table: &mut Vec<Slot>, cap: u64, mut e: Slot) {
    let mut at = home(e[0], cap);
    loop {
        if at + WINDOW >= table.len() {
            table.resize(at + WINDOW + 1, [0; 2]);
        }
        let s = table[at];
        if s[0] < e[0] {
            table[at] = e;
            if s[0] == 0 {
                return;
            }
            e = s;
        }
        at += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::try_pack;

    fn key(text: &[u8]) -> u64 {
        try_pack(text).unwrap()
    }

    #[test]
    fn transitive_components_include_link_only_aliases() {
        let slow = SlowIntern::new();
        let (a, b, c, d) = (key(b"u9"), key(b"u5"), key(b"u1"), key(b"u0"));
        let aliases = build(vec![vec![(a, b)], vec![(c, d), (b, c)]], &slow, 4, 16);
        for linked in [a, b, c, d] {
            assert_eq!(aliases.resolve(linked), d);
        }
    }

    #[test]
    fn case_is_exact_and_bytewise_minimum_wins() {
        let slow = SlowIntern::new();
        let upper = key(b"Uzz");
        let lower = key(b"u0");
        let other = key(b"u1");
        let aliases = build(vec![vec![(lower, other), (other, upper)]], &slow, 3, 16);
        assert_eq!(aliases.resolve(lower), upper);
        assert_eq!(aliases.resolve(other), upper);
        assert_eq!(aliases.resolve(upper), upper);
    }

    #[test]
    fn absent_and_self_linked_ids_are_isolated() {
        let slow = SlowIntern::new();
        let cited = key(b"u2");
        let absent = key(b"u3");
        let aliases = build(vec![vec![(cited, cited)]], &slow, 8, 16);
        assert_eq!(aliases.resolve(cited), cited);
        assert_eq!(aliases.resolve(absent), absent);

        let empty = build(Vec::new(), &slow, 0, 16);
        assert_eq!(empty.resolve(absent), absent);
    }

    /// Clusters longer than the window, and spills past the end that make
    /// the table grow, still resolve exactly.
    #[test]
    fn long_clusters_and_growth() {
        let cap = 64u64;
        let mut table = vec![[0u64; 2]; cap as usize + WINDOW];
        let (mut stored, mut absent) = (Vec::new(), Vec::new());
        let mut k = 1u64 << 60;
        while stored.len() < 40 || absent.len() < 40 {
            k += 1;
            if home(hash(k), cap) + 3 < cap as usize {
                continue;
            }
            if stored.len() < 40 && k % 2 == 0 {
                settle(&mut table, cap, [hash(k), k ^ 0xff]);
                stored.push(k);
            } else if absent.len() < 40 {
                absent.push(k);
            }
        }
        assert!(table.len() > cap as usize + WINDOW);
        let aliases = Aliases { table: table.into_boxed_slice(), cap };
        for &k in &stored {
            assert_eq!(aliases.resolve(k), k ^ 0xff);
        }
        for &k in &absent {
            assert_eq!(aliases.resolve(k), k);
        }
    }

    /// Random forests, cycles, repeats, long chains, upper case and ids too
    /// long to pack, against a plain string union-find. Large enough for
    /// partition spills and lookups past the window.
    #[test]
    fn matches_reference_on_random_graphs() {
        let mut rand = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u64| {
            rand ^= rand << 13;
            rand ^= rand >> 7;
            rand ^= rand << 17;
            rand % n
        };
        for (ids, edges) in [(30usize, 20usize), (5_000, 4_000), (150_000, 120_000)] {
            let mut slow = SlowIntern::new();
            let mut seen = std::collections::HashSet::new();
            let mut names: Vec<Vec<u8>> = Vec::new();
            while names.len() < ids {
                let i = names.len();
                let len = 1 + next(if i % 97 == 0 { 14 } else { 8 }) as usize;
                let mut s = vec![b"ukeUKE"[next(if i % 50 == 0 { 6 } else { 2 }) as usize]];
                s.extend((0..len).map(|_| b"0123456789abcdefghijklmnopqrstuvwxyz"[next(36) as usize]));
                if seen.insert(s.clone()) {
                    names.push(s);
                }
            }
            let keys: Vec<u64> = names.iter().map(|s| try_pack(s).unwrap_or_else(|| slow.intern(s))).collect();
            let mut pairs = Vec::new();
            for _ in 0..edges {
                let a = next(ids as u64) as usize;
                // Mostly local edges make many mid-sized components; one
                // long chain makes a deep one.
                let b = if next(4) == 0 { next(ids as u64) as usize } else { (a + 1 + next(3) as usize) % ids };
                pairs.push((a, b));
            }
            pairs.extend((0..ids.min(400)).map(|i| (i * 7 % ids, (i * 7 + 7) % ids)));
            // Reference: union-find on indices, naming by byte order.
            let mut parent: Vec<usize> = (0..ids).collect();
            fn find(parent: &mut [usize], mut x: usize) -> usize {
                while parent[x] != x {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                x
            }
            for &(a, b) in &pairs {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra.max(rb)] = ra.min(rb);
            }
            let mut best: Vec<Option<usize>> = vec![None; ids];
            let mut cited = vec![false; ids];
            for &(a, b) in &pairs {
                cited[a] = true;
                cited[b] = true;
            }
            for i in (0..ids).filter(|&i| cited[i]) {
                let r = find(&mut parent, i);
                if best[r].is_none_or(|j| names[i] < names[j]) {
                    best[r] = Some(i);
                }
            }
            let input: Vec<(u64, u64)> = pairs.iter().map(|&(a, b)| (keys[a], keys[b])).collect();
            for threads in [1, 3, 8] {
                let parts: Vec<Vec<(u64, u64)>> = input.chunks(input.len().div_ceil(5)).map(<[_]>::to_vec).collect();
                let aliases = build(parts, &slow, threads, 64);
                for i in 0..ids {
                    let want = if cited[i] { keys[best[find(&mut parent, i)].unwrap()] } else { keys[i] };
                    assert_eq!(aliases.resolve(keys[i]), want, "id {i} threads {threads}");
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::{user_key, wire, Shared};
    use std::path::Path;
    use std::time::Instant;

    fn bodies(dir: &Path) -> Vec<Vec<u8>> {
        let mut v: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        v.sort();
        v.into_iter().map(|p| std::fs::read(p).unwrap()).collect()
    }

    /// Process CPU seconds: less disturbed than wall time on a busy machine.
    fn cpu_time() -> f64 {
        #[repr(C)]
        struct Timespec { sec: i64, nsec: i64 }
        unsafe extern "C" { fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32; }
        let clock = if cfg!(target_os = "macos") { 12 } else { 2 }; // CLOCK_PROCESS_CPUTIME_ID
        let mut ts = Timespec { sec: 0, nsec: 0 };
        // SAFETY: `ts` is a valid out-pointer for the libc call.
        unsafe { clock_gettime(clock, &mut ts) };
        ts.sec as f64 + ts.nsec as f64 * 1e-9
    }

    fn body(b: &[u8]) -> &[u8] {
        memchr::memchr(b'\n', b).map_or(&[], |i| &b[i + 1..])
    }

    // cargo test --release --bin solve bench_links -- --ignored --nocapture
    // LINKS_DATA=/tmp/ss-finalshape-73 for the 60M shape.
    #[test]
    #[ignore]
    fn bench_links() {
        let data = std::env::var("LINKS_DATA").unwrap_or_else(|_| "/tmp/ss-20m".into());
        let data = Path::new(&data);
        let shared = Shared { slow: Mutex::new(SlowIntern::new()) };
        let mut parts: Vec<Vec<(u64, u64)>> = (0..8).map(|_| Vec::new()).collect();
        for (i, b) in bodies(&data.join("links")).iter().enumerate() {
            wire::rows::<2>(body(b), |[a, b]| {
                if !a.is_empty() && !b.is_empty() && a != b {
                    parts[i % 8].push((user_key(&shared, a), user_key(&shared, b)));
                }
            });
        }
        let mut queries = Vec::new();
        let resolve = std::env::var_os("LINKS_NORESOLVE").is_none();
        for b in bodies(&data.join("events")).into_iter().filter(|_| resolve) {
            wire::rows::<7>(body(&b), |f| if !f[1].is_empty() { queries.push(user_key(&shared, f[1])) });
        }
        for b in bodies(&data.join("clicks")).into_iter().filter(|_| resolve) {
            wire::rows::<4>(body(&b), |f| if !f[1].is_empty() { queries.push(user_key(&shared, f[1])) });
        }
        let slow = shared.slow.into_inner().unwrap();
        let edges = parts.iter().map(Vec::len).sum::<usize>();
        let user_bins = 512;
        // rename() visits keys bin by bin.
        let shift = 64 - (user_bins as u64).trailing_zeros();
        queries.sort_by_key(|&k| hash(k) >> shift);
        let reps = std::env::var("LINKS_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        let counts = std::env::var("LINKS_THREADS").unwrap_or_else(|_| "1,8".into());
        for threads in counts.split(',').map(|t| t.parse::<usize>().unwrap()) {
            let (mut best, mut cpu) = (f64::MAX, f64::MAX);
            for _ in 0..reps {
                let p = parts.clone();
                let (t, c) = (Instant::now(), cpu_time());
                let a = build(p, &slow, threads, user_bins);
                best = best.min(t.elapsed().as_secs_f64());
                cpu = cpu.min(cpu_time() - c);
                std::hint::black_box(&a);
            }
            eprintln!("build threads={threads}: {:7.2} ms {:6.1} ns/edge, cpu {:7.2} ms", best * 1e3, best * 1e9 / edges as f64, cpu * 1e3);
        }
        if !resolve {
            return;
        }
        let aliases = build(parts.clone(), &slow, 8, user_bins);
        let time = |f: &mut dyn FnMut()| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best * 1e9 / queries.len() as f64
        };
        let renamed = queries.iter().filter(|&&k| aliases.resolve(k) != k).count();
        let lookup = time(&mut || {
            let acc = queries.iter().fold(0u64, |acc, &k| acc ^ aliases.resolve(k));
            std::hint::black_box(acc);
        });
        // rename()'s shape: keep self-named keys in place and scatter renamed
        // ones to their name's bin. Branching on "renamed" per key, versus
        // splitting without that branch and scattering afterwards.
        let mut dest: Vec<Vec<u64>> = (0..user_bins).map(|_| Vec::with_capacity(queries.len() / user_bins)).collect();
        let mut keep = queries.clone();
        let branchy = time(&mut || {
            dest.iter_mut().for_each(Vec::clear);
            keep.copy_from_slice(&queries);
            let mut kept = 0;
            for i in 0..keep.len() {
                let k = keep[i];
                let n = aliases.resolve(k);
                if n == k {
                    keep[kept] = k;
                    kept += 1;
                } else {
                    dest[(hash(n) >> shift) as usize].push(n);
                }
            }
            std::hint::black_box(kept);
        });
        let mut moved = vec![0u64; 1 << 14];
        let split = time(&mut || {
            dest.iter_mut().for_each(Vec::clear);
            keep.copy_from_slice(&queries);
            let mut kept = 0;
            for start in (0..keep.len()).step_by(moved.len()) {
                let mut m = 0;
                for i in start..(start + moved.len()).min(keep.len()) {
                    let k = keep[i];
                    let n = aliases.resolve(k);
                    keep[kept] = k;
                    kept += usize::from(n == k);
                    moved[m] = n;
                    m += usize::from(n != k);
                }
                for &n in &moved[..m] {
                    dest[(hash(n) >> shift) as usize].push(n);
                }
            }
            std::hint::black_box(kept);
        });
        eprintln!("resolve: {lookup:.2} ns/query, {} queries, {:.1}% renamed, {edges} edges", queries.len(), renamed as f64 * 100.0 / queries.len() as f64);
        eprintln!("rename-like: branch per key {branchy:.2} ns/key, branch-free split {split:.2} ns/key");
    }
}
