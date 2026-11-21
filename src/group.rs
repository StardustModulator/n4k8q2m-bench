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

#[derive(Clone, Copy, Default)]
struct Slot { key: u64, ev: u32, cl: u32 }

/// Buffers reused across a worker's bins.
#[derive(Default)]
pub(super) struct Scratch {
    slots: Vec<Slot>,
    ids: Vec<u32>,
    ev: Vec<u64>,
    cl: Vec<Hit>,
}

const MUL: u64 = 0x9e37_79b9_7f4a_7c15;

// Within a person shard the table and cursors fit in private caches: number
// every row by its person's slot, count, then scatter each person's events
// and clicks together. Neither events nor clicks get sorted (see `aggregate`).
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
    // Person numbers go to a reused array rather than into the parts, which
    // stay clean and are only read twice. A table that fills up is doubled
    // and the pass redone; the numbers so far are simply overwritten.
    let ids = &mut sc.ids;
    if ids.len() < ne + nc { ids.resize(ne + nc, 0); }
    let mut slots = std::mem::take(&mut sc.slots);
    while !number(&parts, &mut slots, &mut ids[..ne + nc], bits) {
        slots.fill(Slot::default());
        bits += 1;
    }
    let t1 = Instant::now();
    let (mut es, mut cs) = (0u32, 0u32);
    for p in &mut slots {
        let (e, c) = (p.ev, p.cl);
        p.ev = es; p.cl = cs;
        es += e; cs += c;
    }
    // Every cell below es and cs is written by the scatter; stale contents
    // from earlier bins need no clearing.
    let (ev, cl) = (&mut sc.ev, &mut sc.cl);
    if ev.len() < es as usize { ev.resize(es as usize, 0); }
    if cl.len() < cs as usize { cl.resize(cs as usize, Hit { ts: 0, id: 0, camp: 0 }); }
    debug_assert!(cs as usize <= nc);
    // `number` gave every event a slot below slots.len() and counted it in
    // that slot, so each cursor stays below the next slot's start <= es (cs).
    let (ev_ids, cl_ids) = ids[..ne + nc].split_at(ne);
    // Plain counted loops: an iterator chain over the parts kept the loop
    // body several instructions longer than the work itself.
    let mut n = 0;
    for p in &parts {
        for r in &p.ev {
            // SAFETY: n counts the events, which have one number each.
            let id = unsafe { *ev_ids.get_unchecked(n) } as usize;
            n += 1;
            // SAFETY: see above.
            let cursor = unsafe { &mut slots.get_unchecked_mut(id).ev };
            unsafe { *ev.get_unchecked_mut(*cursor as usize) = r.val };
            *cursor += 1;
        }
    }
    let mut n = 0;
    for p in &parts {
        for r in &p.cl {
            // SAFETY: n counts the clicks, which have one number each.
            let id = unsafe { *cl_ids.get_unchecked(n) } as usize;
            n += 1;
            if id != u32::MAX as usize {
                // SAFETY: see above.
                let cursor = unsafe { &mut slots.get_unchecked_mut(id).cl };
                unsafe { *cl.get_unchecked_mut(*cursor as usize) = r.hit };
                *cursor += 1;
            }
        }
    }
    let t2 = Instant::now();
    let (mut e0, mut c0) = (0, 0);
    for p in &mut slots {
        let (e1, c1) = (p.ev as usize, p.cl as usize);
        if e1 > e0 {
            acc.person(p.key, &mut ev[e0..e1], &mut cl[c0..c1], cats, slow, first_day);
        }
        e0 = e1; c0 = c1;
        // Leave the table empty for the next bin.
        *p = Slot::default();
    }
    sc.slots = slots;
    acc.scratch = sc;
    let t3 = Instant::now();
    #[cfg(test)]
    bench::phase([t0, t1, t2, t3]);
    for (p, (a, b)) in acc.phases.0.iter_mut().zip([(t0, t1), (t1, t2), (t2, t3)]) {
        *p += (b - a).as_nanos() as u64;
    }
}

// Open addressing with linear probing; a slot number is a person's number.
// One spare slot after the last lets the two-slot check skip the wraparound;
// it stays empty. Returns false, with `slots` dirty, once three quarters of
// the slots are taken. `slots` comes in empty.
fn number(parts: &[Histories], slots: &mut Vec<Slot>, ids: &mut [u32], bits: u32) -> bool {
    let mask = (1usize << bits) - 1;
    slots.resize(mask + 2, Slot::default());
    let slots = &mut slots[..];
    let shift = 64 - bits;
    let mut room = mask / 4 * 3;
    let mut n = 0;
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
                }
            }
            // SAFETY: id <= mask + 1 < slots.len(); ids holds one entry per
            // record and n counts them.
            unsafe { slots.get_unchecked_mut(id) }.ev += 1;
            unsafe { *ids.get_unchecked_mut(n) = id as u32 };
            n += 1;
        }
    }
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
            let found = slots[id].key == key;
            if found { slots[id].cl += 1; }
            // SAFETY: as for events.
            unsafe { *ids.get_unchecked_mut(n) = if found { id as u32 } else { u32::MAX } };
            n += 1;
        }
    }
    debug_assert_eq!(n, ids.len());
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
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::sync::Mutex;

    static CAPTURE: Mutex<Option<Vec<Vec<Histories>>>> = Mutex::new(None);
    static PHASE: Mutex<[f64; 3]> = Mutex::new([0.0; 3]);

    pub(super) fn phase(t: [Instant; 4]) {
        let mut p = PHASE.lock().unwrap();
        for i in 0..3 { p[i] += (t[i + 1] - t[i]).as_secs_f64(); }
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
        let mut best_p = [f64::MAX; 3];
        let rounds = std::env::var("BENCH_ROUNDS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
        for round in 0..rounds {
            let copies = copy(&bins);
            *PHASE.lock().unwrap() = [0.0; 3];
            let mut acc = Acc::new(t.camps.len(), days, t.cat_names.len());
            let start = Instant::now();
            for b in copies {
                histories(b, &mut acc, &t.cat_names, &t.slow, w.start_day);
            }
            let el = start.elapsed().as_secs_f64();
            best = best.min(el);
            let p = *PHASE.lock().unwrap();
            for i in 0..3 { best_p[i] = best_p[i].min(p[i]); }
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
        eprintln!("histories: {:.1} ms ({:.2} ns/record) intern={:.1} scatter={:.1} person={:.1}", best * 1e3, best * 1e9 / (ne + nc) as f64,
            best_p[0] * 1e3, best_p[1] * 1e3, best_p[2] * 1e3);
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
