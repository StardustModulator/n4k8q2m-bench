//! Exact linked-id resolution. Link endpoints are interned per hash
//! partition, joined by a lock-free disjoint set whose roots are always their
//! component's bytewise-smallest id, and every id named by another id lands
//! in one flat table laid out in hash order.

use std::cmp::Ordering;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::thread;

use super::stream::hash;
use super::{SlowIntern, cmp_keys};

/// One renamed id. `hash` is `stream::hash(key)`, a bijection with
/// hash(0) == 0: it identifies the key, and zero marks an empty slot.
#[derive(Clone, Copy, Default)]
struct Slot {
    hash: u64,
    name: u64,
}

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
            name = if s.hash == h { s.name } else { name };
        }
        if w[WINDOW - 1].hash > h {
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
            let s = self.table[at];
            if s.hash == h {
                return s.name;
            }
            if s.hash < h {
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
    let mut at = home(e.hash, cap) - offset;
    while at < slots.len() {
        let s = slots[at];
        if s.hash < e.hash {
            slots[at] = e;
            if s.hash == 0 {
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

/// Follow parents with path halving. `link[x]` holds `x ^ parent(x)`, so
/// zero (fresh zeroed memory) is a root. Parents only ever move to smaller
/// keys, so a stale relaxed load still points at an ancestor.
#[inline(always)]
fn root(link: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = x ^ link[x as usize].load(Relaxed);
        if p == x {
            return x;
        }
        let g = p ^ link[p as usize].load(Relaxed);
        if g != p {
            link[x as usize].store(x ^ g, Relaxed);
        }
        x = g;
    }
}

/// Lock-free union. The root with the larger key becomes a child of the
/// other, so every parent has a smaller key than its child: concurrent links
/// cannot form a cycle, and each root is its component's minimum. Returns
/// whether this call linked two roots.
#[inline(always)]
fn unite(link: &[AtomicU32], keys: &[u64], slow: &SlowIntern, mut a: u32, mut b: u32) -> bool {
    loop {
        a = root(link, a);
        b = root(link, b);
        if a == b {
            return false;
        }
        let (lo, hi) = if cmp_keys(keys[a as usize], keys[b as usize], slow) == Ordering::Less {
            (a, b)
        } else {
            (b, a)
        };
        if link[hi as usize].compare_exchange(0, hi ^ lo, Relaxed, Relaxed).is_ok() {
            return true;
        }
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
    let mut keys = Vec::with_capacity(occurrences);
    for bin in bins {
        for x in bin.iter_mut() {
            let k = *x;
            debug_assert_ne!(k, 0);
            let mut at = hash(k) as usize & mask;
            loop {
                let s = seen[at];
                if s == k {
                    *x = u64::from(index[at]);
                    break;
                }
                if s == 0 {
                    seen[at] = k;
                    index[at] = keys.len() as u32;
                    *x = keys.len() as u64;
                    keys.push(k);
                    break;
                }
                at = (at + 1) & mask;
            }
        }
    }
    seen[..cap].fill(0);
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

/// Build an exact resolver from link endpoints only.
pub(super) fn build(
    parts: Vec<Vec<(u64, u64)>>,
    slow: &SlowIntern,
    threads: usize,
    user_bins: usize,
) -> Aliases {
    let _ = user_bins;
    let edge_count = parts.iter().map(Vec::len).sum::<usize>();
    if edge_count == 0 {
        return Aliases { table: vec![Slot::default(); WINDOW].into_boxed_slice(), cap: 0 };
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
    let keys: Vec<u64> = uniq.concat();
    drop(uniq);

    // Replay each worker's edges: the cursors meet the interned indices in
    // the order they were scattered, so no endpoint is looked up again.
    let mut zeroed = vec![0u32; vertices];
    // SAFETY: AtomicU32 has the size and alignment of u32, and `zeroed` is
    // only accessed through this view until the view is dropped.
    let link: &[AtomicU32] = unsafe { std::slice::from_raw_parts(zeroed.as_mut_ptr().cast(), vertices) };
    let renamed = AtomicUsize::new(0);
    thread::scope(|scope| {
        for (span, ids) in spans.iter().zip(&scattered) {
            let (keys, base, renamed) = (&keys, &base, &renamed);
            scope.spawn(move || {
                let mut cursor = vec![0usize; np];
                let mut linked = 0;
                let mut next = |k: u64| {
                    let p = part_of(hash(k), np);
                    let local = ids[p][cursor[p]];
                    cursor[p] += 1;
                    (base[p] + local as usize) as u32
                };
                for slice in span {
                    for &(a, b) in *slice {
                        let (ga, gb) = (next(a), next(b));
                        linked += usize::from(unite(link, keys, slow, ga, gb));
                    }
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
    let mut table = vec![Slot::default(); cap + WINDOW];
    // Partition p holds the hashes whose complement lies in run np - 1 - p.
    let mut runs: Vec<(usize, &mut [Slot])> = table[..cap].chunks_mut(width).enumerate().map(|(q, s)| (np - 1 - q, s)).collect();
    runs.reverse();
    let spilled = each(runs, workers.min(np), Vec::new, |spill, (p, slots)| {
        let offset = (np - 1 - p) * width;
        for g in base[p]..base[p + 1] {
            let r = root(link, g as u32) as usize;
            if r != g {
                let e = Slot { hash: hash(keys[g]), name: keys[r] };
                if let Some(e) = place(slots, offset, cap as u64, e) {
                    spill.push(e);
                }
            }
        }
    });
    drop(zeroed);
    let mut table = table;
    for mut e in spilled.into_iter().flatten() {
        let mut at = home(e.hash, cap as u64);
        loop {
            if at + WINDOW >= table.len() {
                table.resize(at + WINDOW + 1, Slot::default());
            }
            let s = table[at];
            if s.hash < e.hash {
                table[at] = e;
                if s.hash == 0 {
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
        for threads in [1, 8] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let p = parts.clone();
                let t = Instant::now();
                let a = build(p, &slow, threads, user_bins);
                best = best.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&a);
            }
            eprintln!("build threads={threads}: {:7.2} ms {:6.1} ns/edge", best * 1e3, best * 1e9 / edges as f64);
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
