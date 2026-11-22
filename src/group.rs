//! Per-person grouping of a person bin and its hand-off to sessionization.
//!
//! CONTRACT (frozen between modules): `stream` delivers, for every person
//! bin, worker parts of `Histories`. `Event.key`/`Click.key` is the person's
//! name key (bytewise-smallest linked id, packed or interned); every event and
//! click of a person lands in the same bin. `Event.val` is
//! `ts_offset << 32 | category << 24 | price` (price 0 for view/cart).
use super::*;
use crate::aggregate::{Acc, Hit};

#[derive(Clone, Copy)]
pub(super) struct Event {
    pub(super) key: u64,
    pub(super) val: u64,
}

#[derive(Clone, Copy)]
pub(super) struct Click {
    pub(super) key: u64,
    pub(super) hit: Hit,
}

#[derive(Default)]
pub(super) struct Histories {
    pub(super) ev: Vec<Event>,
    pub(super) cl: Vec<Click>,
}

/// A person's records are chained in chunks of one cache line: seven values
/// and, in the last position, the index of the chunk written before it, 0 at
/// the start of the chain. A slot keeps only the write cursor; the cursor's
/// chunk holds `cursor & 7` values, every earlier one is full. Chunk 0 is
/// never written: an empty slot's cursor sits on its link position, so the
/// first record opens a chunk like a full one does, linked to 0.
const CHUNK: usize = 8;
const FULL: u32 = CHUNK as u32 - 1;

pub(super) trait Link: Copy {
    fn link(self) -> usize;
    fn mark(at: usize) -> Self;
}

impl Link for u64 {
    #[inline(always)]
    fn link(self) -> usize { self as usize }
    #[inline(always)]
    fn mark(at: usize) -> u64 { at as u64 }
}

impl Link for Hit {
    #[inline(always)]
    fn link(self) -> usize { self.id as usize }
    #[inline(always)]
    fn mark(at: usize) -> Hit { Hit { id: at as u64, ts: 0, camp: 0 } }
}

/// One person's records, read from the newest chunk back. The order of a
/// person's records never matters to `aggregate`.
#[derive(Clone, Copy)]
pub(super) struct Chain<'a, T> {
    arena: &'a [T],
    cur: u32,
}

impl<'a, T: Link> Chain<'a, T> {
    #[inline(always)]
    pub(super) fn is_empty(&self) -> bool { self.cur == FULL }

    /// The newest chunk's values and the index of the one before it.
    #[inline(always)]
    pub(super) fn last(&self) -> (&'a [T], usize) {
        let cur = self.cur as usize;
        let base = cur & !(CHUNK - 1);
        (&self.arena[base..cur], self.arena[base + CHUNK - 1].link())
    }

    /// The newest chunk's whole block of slots, how many hold values (the
    /// rest are stale, then the link) and the index of the chunk before it.
    #[inline(always)]
    pub(super) fn last_block(&self) -> (&'a [T; CHUNK], usize, usize) {
        let cur = self.cur as usize;
        let base = cur & !(CHUNK - 1);
        let block: &[T; CHUNK] = self.arena[base..base + CHUNK].try_into().unwrap();
        (block, cur - base, block[CHUNK - 1].link())
    }

    /// The values of the full chunk at `at` and the index of the one before it.
    #[inline(always)]
    pub(super) fn full(&self, at: usize) -> (&'a [T], usize) {
        (&self.arena[at..at + CHUNK - 1], self.arena[at + CHUNK - 1].link())
    }

    /// Every chunk's values, newest first. Not for an empty chain.
    #[inline(always)]
    pub(super) fn for_each_chunk(&self, mut f: impl FnMut(&[T])) {
        let (mut vals, mut prev) = self.last();
        loop {
            f(vals);
            if prev == 0 { return; }
            (vals, prev) = self.full(prev);
        }
    }

    #[inline(always)]
    pub(super) fn for_each(&self, mut f: impl FnMut(&T)) {
        self.for_each_chunk(|c| c.iter().for_each(&mut f));
    }

    /// Build one from values, for tests and the long-history copies.
    #[cfg(test)]
    pub(super) fn build(vals: &[T], arena: &'a mut Vec<T>) -> Self {
        arena.clear();
        // Room for the chain plus the free chunk every append links.
        arena.resize(CHUNK * (vals.len() / (CHUNK - 1) + 3), T::mark(0));
        let (mut top, mut cur) = (CHUNK as u32, FULL);
        for &v in vals {
            append(arena, &mut top, &mut cur, v);
        }
        Chain { arena, cur }
    }
}

/// Write one record at a slot's cursor, opening a chunk when it is full.
/// The arena holds at least `top + CHUNK` entries (see the bound in `number`).
/// No branch depends on the data: the link of the next free chunk is written
/// every time, and holds the right value at the moment that chunk is opened.
#[inline(always)]
fn append<T: Link>(arena: &mut [T], top: &mut u32, cur: &mut u32, val: T) {
    let c = *cur;
    let full = c & FULL == FULL;
    // SAFETY: top + CHUNK <= arena.len() by the caller's bound.
    unsafe { *arena.get_unchecked_mut(*top as usize + CHUNK - 1) = T::mark((c - FULL) as usize) };
    let at = if full { *top } else { c };
    *top += if full { CHUNK as u32 } else { 0 };
    // SAFETY: at < top <= arena.len() - CHUNK.
    unsafe { *arena.get_unchecked_mut(at as usize) = val };
    *cur = at + 1;
}

#[derive(Clone, Copy)]
struct Slot { key: u64, ev: u32, cl: u32 }

const EMPTY: Slot = Slot { key: 0, ev: FULL, cl: FULL };

/// Buffers reused across a worker's bins.
#[derive(Default)]
pub(super) struct Scratch {
    slots: Vec<Slot>,
    // Occupied slots in order of first appearance.
    persons: Vec<u32>,
    ev: Vec<u64>,
    cl: Vec<Hit>,
}

const MUL: u64 = 0x9e37_79b9_7f4a_7c15;
// Within a person shard the table and the chunk arenas fit in private
// caches: one pass chains every row onto its person's slot, then each person
// is summarized from its chain. Neither events nor clicks get sorted (see
// `aggregate`).
pub(super) fn histories(parts: Vec<Histories>, acc: &mut Acc, cats: &[Vec<u8>], slow: &SlowIntern, first_day: i32) {
    #[cfg(test)]
    bench::capture(&parts);
    let ne = parts.iter().map(|p| p.ev.len()).sum::<usize>();
    if ne == 0 { return; }
    let nc = parts.iter().map(|p| p.cl.len()).sum::<usize>();
    // Rows grow by a large step so the vector rarely moves; a bin yields a
    // small fraction of its events in rows, and untouched capacity costs no
    // pages.
    if acc.rows.capacity() - acc.rows.len() < ne {
        acc.rows.reserve(ne * 64);
    }
    let t0 = Instant::now();
    let mut bits = (ne / 8).max(64).next_power_of_two().trailing_zeros();
    let mut sc = std::mem::take(&mut acc.scratch);
    // A table that fills up is doubled and the pass redone from the start.
    while !number(&parts, &mut sc, ne, nc, bits) {
        sc.slots.fill(EMPTY);
        sc.persons.clear();
        bits += 1;
    }
    let t1 = Instant::now();
    for &id in &sc.persons {
        let id = id as usize;
        let s = sc.slots[id];
        acc.person(s.key, Chain { arena: &sc.ev, cur: s.ev }, Chain { arena: &sc.cl, cur: s.cl }, cats, slow, first_day);
        // Leave the table empty for the next bin.
        sc.slots[id] = EMPTY;
    }
    sc.persons.clear();
    acc.scratch = sc;
    let t2 = Instant::now();
    #[cfg(test)]
    bench::phase([t0, t1, t2]);
    for (p, (a, b)) in acc.phases.0.iter_mut().zip([(t0, t1), (t1, t2)]) {
        *p += (b - a).as_nanos() as u64;
    }
}

// Open addressing with linear probing. One spare slot after the last lets
// the two-slot check skip the wraparound; it stays empty. Returns false, with
// the scratch dirty, once three quarters of the slots are taken. `sc` comes
// in with every slot EMPTY and no persons.
#[inline(never)]
fn number(parts: &[Histories], sc: &mut Scratch, ne: usize, nc: usize, bits: u32) -> bool {
    let mask = (1usize << bits) - 1;
    sc.slots.resize(mask + 2, EMPTY);
    let mut room = mask / 4 * 3;
    // Every person opens one chunk and then one more per seven records, so
    // the arenas never run out at these sizes (chunk 0 is the end mark).
    let bound = |n: usize| CHUNK * (2 + room.min(n) + n / (CHUNK - 1));
    if sc.ev.len() < bound(ne) { sc.ev.resize(bound(ne), 0); }
    if sc.cl.len() < bound(nc) { sc.cl.resize(bound(nc), Hit { id: 0, ts: 0, camp: 0 }); }
    let slots = &mut sc.slots[..];
    let shift = 64 - bits;
    let mut top = CHUNK as u32;
    for p in parts {
        for e in &p.ev {
            let key = e.key;
            let s = (key.wrapping_mul(MUL) >> shift) as usize;
            // A key sits in its home slot or the next one unless its run is
            // long; only those and new keys take the probing path.
            // SAFETY: s <= mask and s + 1 <= mask + 1 < slots.len().
            let mut id = s + usize::from(unsafe { slots.get_unchecked(s) }.key != key);
            // SAFETY: as above; probe returns a slot <= mask.
            if unsafe { slots.get_unchecked(id) }.key != key {
                id = probe(slots, key, s, mask);
                if slots[id].key == 0 {
                    if room == 0 { return false; }
                    room -= 1;
                    slots[id].key = key;
                    sc.persons.push(id as u32);
                }
            }
            // SAFETY: id <= mask + 1 < slots.len().
            let slot = unsafe { slots.get_unchecked_mut(id) };
            append(&mut sc.ev, &mut top, &mut slot.ev, e.val);
        }
    }
    let mut top = CHUNK as u32;
    for p in parts {
        for c in &p.cl {
            let key = c.key;
            let s = (key.wrapping_mul(MUL) >> shift) as usize;
            // SAFETY: as for events.
            let mut id = s + usize::from(unsafe { slots.get_unchecked(s) }.key != key);
            if unsafe { slots.get_unchecked(id) }.key != key {
                id = probe(slots, key, s, mask);
            }
            // A person without events has no sessions to credit.
            let slot = &mut slots[id];
            if slot.key == key {
                append(&mut sc.cl, &mut top, &mut slot.cl, c.hit);
            }
        }
    }
    true
}

/// The slot holding `key`, or the empty slot that ends its run.
#[cold]
#[inline(never)]
fn probe(slots: &[Slot], key: u64, mut s: usize, mask: usize) -> usize {
    loop {
        let k = slots[s].key;
        if k == key || k == 0 { return s; }
        s = (s + 1) & mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One event per person makes the table double three times. People are
    // split over parts; a few have clicks, and one click has no events.
    #[test]
    fn many_people_restart_the_table() {
        let mut slow = SlowIntern::new();
        let week = WEEK_MS as u32;
        let n = 3000;
        let key = |i: u32| try_pack(format!("u{i}").as_bytes()).unwrap();
        let mut parts: Vec<Histories> = (0..3).map(|_| Histories::default()).collect();
        for i in 0..n {
            let ev = Event { key: key(i), val: u64::from(week + i * 1000) << 32 };
            parts[i as usize % 3].ev.push(ev);
        }
        let mut click = |p, ts, camp, id: &[u8]| {
            let id = try_pack(id).unwrap_or_else(|| slow.intern(id));
            Click { key: key(p), hit: Hit { id, ts, camp } }
        };
        let cl = [
            click(7, week + 7000, 1, b"k7"),                    // At the start.
            click(8, week + 7999, 0, b"a-long-interned-id-8"), // Just before it.
            click(9, 8999, 1, b"k9"),                           // A week and 1 ms before.
            click(n + 5, week, 1, b"korphan"),                  // No events.
        ];
        parts[1].cl.extend(&cl[..2]);
        parts[2].cl.extend(&cl[2..]);
        let cats = vec![b"c".to_vec()];
        let mut acc = Acc::new(3, 1, cats.len());
        histories(parts, &mut acc, &cats, &slow, 0);
        let mut keys: Vec<u64> = acc.rows.iter().map(|r| r.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n as usize);
        assert_eq!(acc.rows.len(), n as usize);
        assert!(acc.rows.iter().all(|r| (r.day, r.sessions, r.events, r.revenue, r.top) == (0, 1, 1, 0, u16::MAX)));
        assert_eq!(acc.stats, [[1, 0, 0, 1], [1, 0, 0, 1], [i64::from(n) - 2, 0, 0, 0]]);
    }

    // Chains hand back every value exactly once, newest chunk first, and
    // the arena bound of `number` holds for persons of every size.
    #[test]
    fn chains_keep_every_record() {
        for n in 0..40usize {
            let vals: Vec<u64> = (1..=n as u64).collect();
            let mut arena = Vec::new();
            let chain = Chain::build(&vals, &mut arena);
            assert_eq!(chain.is_empty(), n == 0);
            if n == 0 { continue; }
            let mut got = Vec::new();
            let mut chunks = 0;
            chain.for_each_chunk(|c| { chunks += 1; got.extend_from_slice(c); });
            assert_eq!(chunks, n.div_ceil(CHUNK - 1));
            got.sort_unstable();
            assert_eq!(got, vals);
        }
        // Many persons with few events each, split over parts, through
        // `histories` itself: every event is summarized exactly once.
        let cats = vec![b"c".to_vec()];
        let slow = SlowIntern::new();
        let week = WEEK_MS as u32;
        for per in [1usize, 6, 7, 8, 13, 14, 15, 29] {
            let persons = 500;
            let mut parts: Vec<Histories> = (0..4).map(|_| Histories::default()).collect();
            for p in 0..persons as u32 {
                for e in 0..per as u32 {
                    // A person's events are more than a gap apart: one session each.
                    let ts = week + e * (GAP_MS as u32 + 1) + p * 1000;
                    parts[(p + e) as usize % 4].ev.push(Event { key: u64::from(p) + 1, val: u64::from(ts) << 32 });
                }
            }
            let mut acc = Acc::new(1, 40, cats.len());
            histories(parts, &mut acc, &cats, &slow, 0);
            let (sessions, events): (i64, i64) = acc.rows.iter().fold((0, 0), |a, r| (a.0 + r.sessions, a.1 + r.events));
            assert_eq!((sessions, events), ((persons * per) as i64, (persons * per) as i64), "per={per}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::sync::Mutex;

    static CAPTURE: Mutex<Option<Vec<Vec<Histories>>>> = Mutex::new(None);
    static PHASE: Mutex<[f64; 2]> = Mutex::new([0.0; 2]);

    pub(super) fn phase(t: [Instant; 3]) {
        let mut p = PHASE.lock().unwrap();
        for i in 0..2 { p[i] += (t[i + 1] - t[i]).as_secs_f64(); }
    }

    pub(super) fn capture(parts: &[Histories]) {
        if let Some(c) = CAPTURE.lock().unwrap().as_mut() {
            c.push(parts.iter().map(|p| Histories { ev: p.ev.clone(), cl: p.cl.clone() }).collect());
        }
    }

    fn copy(bins: &[Vec<Histories>]) -> Vec<Vec<Histories>> {
        bins.iter().map(|b| b.iter().map(|p| Histories { ev: p.ev.clone(), cl: p.cl.clone() }).collect()).collect()
    }

    // BENCH_DATA=/tmp/ss-20m cargo test --release --bin solve bench_group -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_group() {
        let data = std::env::var("BENCH_DATA").unwrap_or_else(|_| "/tmp/ss-20m".into());
        let data = Path::new(&data);
        *CAPTURE.lock().unwrap() = Some(Vec::new());
        let t = crate::stream::compute(data, 8).unwrap();
        let bins = CAPTURE.lock().unwrap().take().unwrap();
        let w = load_window(&data.join("manifest.json")).unwrap();
        let days = ((w.end - 1).div_euclid(DAY_MS) - w.start_day as i64 + 1) as usize;
        let mut want: Vec<_> = t.users.iter().map(|r| (r.key, r.day, r.sessions, r.events, r.revenue, r.top)).collect();
        want.sort_unstable();
        let ne: usize = bins.iter().flatten().map(|p| p.ev.len()).sum();
        let nc: usize = bins.iter().flatten().map(|p| p.cl.len()).sum();
        eprintln!("bins={} events={ne} clicks={nc} rows={}", bins.len(), want.len());
        let mut best = f64::MAX;
        let mut best_p = [f64::MAX; 2];
        let rounds = std::env::var("BENCH_ROUNDS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
        for round in 0..rounds {
            let copies = copy(&bins);
            *PHASE.lock().unwrap() = [0.0; 2];
            let mut acc = Acc::new(t.camps.len(), days, t.cat_names.len());
            let start = Instant::now();
            for b in copies {
                histories(b, &mut acc, &t.cat_names, &t.slow, w.start_day);
            }
            let el = start.elapsed().as_secs_f64();
            best = best.min(el);
            let p = *PHASE.lock().unwrap();
            for i in 0..2 { best_p[i] = best_p[i].min(p[i]); }
            if round == 0 && std::env::var_os("BENCH_NOCHECK").is_none() {
                let mut got: Vec<_> = acc.rows.iter().map(|r| (r.key, r.day, r.sessions, r.events, r.revenue, r.top)).collect();
                got.sort_unstable();
                assert!(got == want, "user rows differ");
                let mut camps = Vec::new();
                for (i, s) in acc.stats.iter().enumerate() {
                    if s[0] != 0 || s[3] != 0 {
                        camps.push(((i / days) as u16, w.start_day + (i % days) as i32, s[0], s[1], s[2], s[3]));
                    }
                }
                let want_c: Vec<_> = t.camps_out.iter().map(|r| (r.camp, r.day, r.sessions, r.purchasing, r.revenue, r.assisted)).collect();
                assert!(camps == want_c, "campaign rows differ");
            }
        }
        eprintln!("histories: {:.1} ms ({:.2} ns/record) chain={:.1} person={:.1}", best * 1e3, best * 1e9 / (ne + nc) as f64,
            best_p[0] * 1e3, best_p[1] * 1e3);
        if std::env::var_os("BENCH_STATS").is_some() {
            stats(&bins);
        }
    }

    fn stats(bins: &[Vec<Histories>]) {
        let mut hist_p = [0usize; 24];
        let mut hist_e = [0usize; 24];
        let mut hist_cp = [0usize; 24];
        let mut hist_cc = [0usize; 24];
        let (mut persons, mut sessions, mut orphan_clicks) = (0usize, 0usize, 0usize);
        let (mut buckets, mut words, mut daysp) = (0usize, 0usize, 0usize);
        let (mut long_p, mut long_e, mut long_c, mut long_s, mut mid_p) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut nsh = [(0usize, 0usize, 0usize); 4];
        let mut per_bin_persons = Vec::new();
        let mut max_e = 0;
        for b in bins {
            let mut m: HashMap<u64, (Vec<u32>, usize)> = HashMap::new();
            for p in b { for e in &p.ev { m.entry(e.key).or_default().0.push((e.val >> 32) as u32); } }
            for p in b { for c in &p.cl { if let Some(x) = m.get_mut(&c.key) { x.1 += 1; } else { orphan_clicks += 1; } } }
            per_bin_persons.push(m.len());
            for (_, (mut ts, ncl)) in m {
                persons += 1;
                let n = ts.len();
                max_e = max_e.max(n);
                let l = (usize::BITS - n.leading_zeros()) as usize;
                hist_p[l] += 1;
                hist_e[l] += n;
                let lc = (usize::BITS - ncl.leading_zeros()) as usize;
                hist_cp[lc] += 1;
                hist_cc[lc] += ncl;
                ts.sort_unstable();
                sessions += 1 + ts.windows(2).filter(|w| w[1] - w[0] > 1_800_000).count();
                let mut bk: Vec<u32> = ts.iter().map(|t| t / 1_800_000).collect();
                bk.dedup();
                buckets += bk.len();
                words += (bk[bk.len() - 1] / 64 - bk[0] / 64) as usize + 1;
                let ns = 1 + ts.windows(2).filter(|w| w[1] - w[0] > 1_800_000).count();
                if ns > 64 { long_p += 1; long_e += ts.len(); long_c += ncl; long_s += ns; }
                let cls = if ns <= 64 { 0 } else if ns <= 128 { 1 } else if ns <= 256 { 2 } else { 3 };
                nsh[cls].0 += 1; nsh[cls].1 += ncl; nsh[cls].2 += ns;
                if ns > 8 { mid_p += 1; }
                let mut dd: Vec<u32> = ts.iter().map(|t| t / 86_400_000).collect();
                dd.dedup();
                daysp += dd.len();
            }
        }
        eprintln!("persons={persons} sessions={sessions} orphan_clicks={orphan_clicks} max_events={max_e} buckets={buckets} words={words} event_days={daysp}");
        eprintln!("long(ns>64): persons={long_p} events={long_e} clicks={long_c} sessions={long_s}; persons with ns>8: {mid_p}");
        eprintln!("ns classes (<=64, <=128, <=256, >256): persons/clicks/sessions {:?}", nsh);
        per_bin_persons.sort_unstable();
        eprintln!("persons/bin min={} med={} max={}", per_bin_persons[0], per_bin_persons[per_bin_persons.len() / 2], per_bin_persons[per_bin_persons.len() - 1]);
        for l in 0..24 {
            if hist_p[l] + hist_cp[l] > 0 {
                eprintln!("  n<{:>8}: persons {:>8} events {:>9} | clicks: persons {:>8} clicks {:>8}", 1usize << l, hist_p[l], hist_e[l], hist_cp[l], hist_cc[l]);
            }
        }
    }
}
