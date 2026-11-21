//! Exact linked-id resolution. Link endpoints are interned per hash
//! partition, joined by a lock-free disjoint set whose roots are always their
//! component's bytewise-smallest id, and every id named by another id lands
//! in one flat table laid out in hash order.

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

/// Find the root and its key. Nearly every vertex is a root or a root's
/// child, which this resolves without a data-dependent branch.
#[inline(always)]
fn root(node: &[AtomicU64], x: u32) -> (u32, u64) {
    let v = node[x as usize].load(Relaxed);
    let p = std::hint::select_unpredictable(is_key(v), x, v as u32);
    let w = node[p as usize].load(Relaxed);
    if is_key(w) {
        return (p, w);
    }
    deep(node, p)
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
/// whether this call linked two roots.
#[inline(always)]
fn unite(node: &[AtomicU64], slow: &SlowIntern, a: u32, b: u32) -> bool {
    let (mut ra, mut ka) = root(node, a);
    let (mut rb, mut kb) = root(node, b);
    loop {
        if ra == rb {
            return false;
        }
        let a_first = cmp_keys(ka, kb, slow) == Ordering::Less;
        let (lo, hi, key) = std::hint::select_unpredictable(a_first, (ra, rb, kb), (rb, ra, ka));
        if node[hi as usize].compare_exchange(key, u64::from(lo), Relaxed, Relaxed).is_ok() {
            return true;
        }
        (ra, ka) = root(node, ra);
        (rb, kb) = root(node, rb);
    }
}

/// Intern one partition's endpoint occurrences in place: each key becomes
/// its partition-local index. Returns the distinct keys in index order.
fn intern(bins: &mut [Vec<u64>], seen: &mut Vec<u64>, index: &mut Vec<u32>) -> Vec<u64> {
    let occurrences = bins.iter().map(Vec::len).sum::<usize>();
    let cap = (occurrences * 2).next_power_of_two().max(16);
    if seen.len() < cap {
        seen.resize(cap, 0);
        index.resize(cap, 0);
    }
    let mask = cap - 1;
    let mut keys = vec![0u64; occurrences];
    let mut n = 0;
    for bin in bins {
        for x in bin.iter_mut() {
            let k = *x;
            assert!(is_key(k), "id key collides with a vertex index");
            let mut at = hash(k) as usize & mask;
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
            *x = u64::from(id);
        }
    }
    seen[..cap].fill(0);
    keys.truncate(n);
    keys
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

    // Each worker owns one contiguous run of the edges, cut across parts.
    let per = edge_count.div_ceil(workers);
    let mut spans: Vec<Vec<&[(u64, u64)]>> = (0..workers).map(|_| Vec::new()).collect();
    let (mut w, mut room) = (0, per);
    for part in &parts {
        let mut rest = &part[..];
        while !rest.is_empty() {
            let take = rest.len().min(room);
            spans[w].push(&rest[..take]);
            rest = &rest[take..];
            room -= take;
            if room == 0 && w + 1 < workers {
                w += 1;
                room = per;
            }
        }
    }
    let workers = spans.len();

    // Scatter endpoint occurrences by partition, in edge order.
    let hint = 2 * per / np + 16;
    let mut scattered: Vec<Vec<Vec<u64>>> = thread::scope(|scope| {
        let handles: Vec<_> = spans
            .iter()
            .map(|span| {
                scope.spawn(move || {
                    let mut out: Vec<Vec<u64>> = (0..np).map(|_| Vec::with_capacity(hint)).collect();
                    for slice in span {
                        for &(a, b) in *slice {
                            debug_assert!(a != 0 && b != 0);
                            out[part_of(hash(a), np)].push(a);
                            out[part_of(hash(b), np)].push(b);
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Intern each partition with one reusable scratch table per worker.
    let jobs: Vec<(usize, Vec<Vec<u64>>)> = (0..np)
        .map(|p| (p, scattered.iter_mut().map(|bins| std::mem::take(&mut bins[p])).collect()))
        .collect();
    let done = each(jobs, workers.min(np), || (Vec::new(), Vec::new(), Vec::new()), |(seen, index, out), (p, mut bins)| {
        let keys = intern(&mut bins, seen, index);
        out.push((p, keys, bins));
    });
    let mut uniq: Vec<Vec<u64>> = (0..np).map(|_| Vec::new()).collect();
    for (_, _, out) in done {
        for (p, keys, bins) in out {
            uniq[p] = keys;
            for (w, bin) in bins.into_iter().enumerate() {
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
    // share of them. Then it replays its own edges: the cursors meet the
    // interned indices in the order they were scattered, so no endpoint is
    // looked up again.
    let mut cells = vec![0u64; vertices];
    const _: () = assert!(std::mem::align_of::<AtomicU64>() == std::mem::align_of::<u64>());
    // SAFETY: AtomicU64 has the size and (asserted) alignment of u64, and
    // `cells` is only accessed through this view until the view is dead.
    let node: &[AtomicU64] = unsafe { std::slice::from_raw_parts(cells.as_mut_ptr().cast(), vertices) };
    let renamed = AtomicUsize::new(0);
    let ready = Barrier::new(workers);
    thread::scope(|scope| {
        for (w, (span, ids)) in spans.iter().zip(&scattered).enumerate() {
            let (base, uniq, renamed, ready) = (&base, &uniq, &renamed, &ready);
            scope.spawn(move || {
                for p in w * np / workers..(w + 1) * np / workers {
                    for (cell, &k) in node[base[p]..base[p + 1]].iter().zip(&uniq[p]) {
                        cell.store(k, Relaxed);
                    }
                }
                ready.wait();
                let mut cursor = vec![0usize; np];
                let mut next = |k: u64| {
                    let p = part_of(hash(k), np);
                    let local = ids[p][cursor[p]];
                    cursor[p] += 1;
                    (base[p] + local as usize) as u32
                };
                // A short software pipeline: resolve ids and prefetch both
                // nodes LAG edges before they are joined.
                const LAG: usize = 16;
                let mut ring = [(0u32, 0u32); LAG];
                let (mut n, mut linked) = (0usize, 0usize);
                for slice in span {
                    for &(a, b) in *slice {
                        let (ga, gb) = (next(a), next(b));
                        prefetch(&node[ga as usize]);
                        prefetch(&node[gb as usize]);
                        let slot = &mut ring[n % LAG];
                        if n >= LAG {
                            linked += usize::from(unite(node, slow, slot.0, slot.1));
                        }
                        *slot = (ga, gb);
                        n += 1;
                    }
                }
                for i in n.saturating_sub(LAG)..n {
                    let (ga, gb) = ring[i % LAG];
                    linked += usize::from(unite(node, slow, ga, gb));
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
    let mut table = vec![[0u64; 2]; cap + WINDOW];
    // Partition p holds the hashes whose complement lies in run np - 1 - p.
    let mut runs: Vec<(usize, &mut [Slot])> = table[..cap].chunks_mut(width).enumerate().map(|(q, s)| (np - 1 - q, s)).collect();
    runs.reverse();
    let spilled = each(runs, workers.min(np), || (Vec::new(), Vec::new()), |(spill, found), (p, slots)| {
        let offset = (np - 1 - p) * width;
        let keys = &uniq[p];
        let nodes = &node[base[p]..base[p + 1]];
        // Gather renamed ids without branching on which they are, fetching
        // each vertex's parent LAG vertices ahead.
        const LAG: usize = 16;
        found.resize(keys.len(), [0; 2]);
        let mut n = 0;
        for (local, &k) in keys.iter().enumerate() {
            if let Some(ahead) = nodes.get(local + LAG) {
                let v = ahead.load(Relaxed);
                prefetch(&node[std::hint::select_unpredictable(is_key(v), base[p] + local + LAG, v as usize)]);
            }
            let g = (base[p] + local) as u32;
            let (r, name) = root(node, g);
            found[n] = [hash(k), name];
            n += usize::from(r != g);
        }
        for &e in &found[..n] {
            if let Some(e) = place(slots, offset, cap as u64, e) {
                spill.push(e);
            }
        }
    });
    let spilled: Vec<Slot> = spilled.into_iter().flat_map(|(spill, _)| spill).collect();
    drop(cells);
    for mut e in spilled {
        let mut at = home(e[0], cap as u64);
        loop {
            if at + WINDOW >= table.len() {
                table.resize(at + WINDOW + 1, [0; 2]);
            }
            let s = table[at];
            if s[0] < e[0] {
                table[at] = e;
                if s[0] == 0 {
                    break;
                }
                e = s;
            }
            at += 1;
        }
    }
    Aliases { table: table.into_boxed_slice(), cap: cap as u64 }
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
        for b in bodies(&data.join("events")) {
            wire::rows::<7>(body(&b), |f| if !f[1].is_empty() { queries.push(user_key(&shared, f[1])) });
        }
        for b in bodies(&data.join("clicks")) {
            wire::rows::<4>(body(&b), |f| if !f[1].is_empty() { queries.push(user_key(&shared, f[1])) });
        }
        let slow = shared.slow.into_inner().unwrap();
        let edges = parts.iter().map(Vec::len).sum::<usize>();
        let user_bins = 512;
        // rename() visits keys bin by bin.
        let shift = 64 - (user_bins as u64).trailing_zeros();
        queries.sort_by_key(|&k| hash(k) >> shift);
        let reps = std::env::var("LINKS_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        for threads in [1, 8] {
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
        let aliases = build(parts.clone(), &slow, 8, user_bins);
        let mut best = f64::MAX;
        let mut renamed = 0;
        for _ in 0..5 {
            let t = Instant::now();
            let mut acc = 0u64;
            renamed = 0;
            for &k in &queries {
                let n = aliases.resolve(k);
                renamed += usize::from(n != k);
                acc ^= n;
            }
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(acc);
        }
        eprintln!("resolve: {:6.2} ns/query, {} queries, {:.1}% renamed, {edges} edges",
            best * 1e9 / queries.len() as f64, queries.len(), renamed as f64 * 100.0 / queries.len() as f64);
    }
}
