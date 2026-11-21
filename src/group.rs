//! Per-person grouping of a person bin and its hand-off to sessionization.
//!
//! CONTRACT (frozen between modules): `stream` delivers, for every person
//! bin, worker parts of `Histories`. `Event.key`/`Click.key` is the person's
//! name key (bytewise-smallest linked id, packed or interned); every event and
//! click of a person lands in the same bin. `Event.val` is
//! `ts_offset << 32 | category << 24 | price` (price 0 for view/cart).
use super::*;
use crate::aggregate::{Acc, Hit};
use std::collections::HashMap;

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

const MUL: u64 = 0x9e37_79b9_7f4a_7c15;

// Within a person shard the table and cursors fit in private caches. A
// person's number is its slot in an open-addressing table that never grows:
// past three quarters full, new people go to an overflow map instead.
pub(super) fn histories(parts: Vec<Histories>, acc: &mut Acc, cats: &[Vec<u8>], slow: &SlowIntern, first_day: i32) {
    #[cfg(test)]
    bench::capture(&parts);
    let mut parts = parts;
    let ne = parts.iter().map(|p| p.ev.len()).sum::<usize>();
    if ne == 0 { return; }
    let nc = parts.iter().map(|p| p.cl.len()).sum::<usize>();
    #[cfg(test)]
    let t0 = Instant::now();
    let bits = (ne / 12).max(64).next_power_of_two().trailing_zeros();
    let mut t = Table::new(bits);
    for part in &mut parts {
        for e in &mut part.ev {
            let key = e.key;
            let s = t.home(key);
            // A key sits in its home slot or the next one unless its run is
            // long; only those and new keys take the probing path.
            let mut id = s + usize::from(t.slots[s].key != key);
            if t.slots[id].key != key {
                id = t.insert(key, s);
            }
            t.slots[id].ev += 1;
            e.key = id as u64;
        }
    }
    for part in &mut parts {
        for c in &mut part.cl {
            let key = c.key;
            let s = t.home(key);
            let mut id = s + usize::from(t.slots[s].key != key);
            if t.slots[id].key != key {
                // A person without events has no sessions to credit.
                id = t.find(key, s);
            }
            if id != usize::MAX { t.slots[id].cl += 1; }
            c.key = id as u64;
        }
    }
    let mut slots = t.slots;
    #[cfg(test)]
    let t1 = Instant::now();
    let (mut es, mut cs) = (0u32, 0u32);
    for p in &mut slots {
        let (e, c) = (p.ev, p.cl);
        p.ev = es; p.cl = cs;
        es += e; cs += c;
    }
    let mut ev = vec![0u64; es as usize];
    let mut cl = vec![Hit { ts: 0, id: 0, camp: 0 }; cs as usize];
    debug_assert!(cl.len() <= nc);
    for part in parts {
        for r in part.ev {
            let cursor = &mut slots[r.key as usize].ev;
            ev[*cursor as usize] = r.val;
            *cursor += 1;
        }
        for r in part.cl {
            if r.key != usize::MAX as u64 {
                let cursor = &mut slots[r.key as usize].cl;
                cl[*cursor as usize] = r.hit;
                *cursor += 1;
            }
        }
    }
    #[cfg(test)]
    let t2 = Instant::now();
    let (mut e0, mut c0) = (0, 0);
    for p in slots {
        let (e1, c1) = (p.ev as usize, p.cl as usize);
        if e1 > e0 {
            acc.person(p.key, &mut ev[e0..e1], &mut cl[c0..c1], cats, slow, first_day);
        }
        e0 = e1; c0 = c1;
    }
    #[cfg(test)]
    bench::phase([t0, t1, t2, Instant::now()]);
}

// Open addressing with linear probing; a slot number is a person's number.
// The table never grows: past three quarters full, new keys get numbers
// after the table from an overflow map. One spare slot after the last lets
// the two-slot check skip the wraparound; it stays empty.
struct Table {
    slots: Vec<Slot>,
    extra: HashMap<u64, u32>,
    mask: usize,
    shift: u32,
    room: usize,
}

impl Table {
    fn new(bits: u32) -> Self {
        let mask = (1usize << bits) - 1;
        Self { slots: vec![Slot::default(); mask + 2], extra: HashMap::new(), mask, shift: 64 - bits, room: mask / 4 * 3 }
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(MUL) >> self.shift) as usize
    }

    #[cold]
    #[inline(never)]
    fn insert(&mut self, key: u64, mut s: usize) -> usize {
        loop {
            let k = self.slots[s].key;
            if k == key { return s; }
            if k == 0 { break; }
            s = (s + 1) & self.mask;
        }
        if self.room > 0 {
            self.room -= 1;
            self.slots[s].key = key;
            return s;
        }
        let slots = &mut self.slots;
        *self.extra.entry(key).or_insert_with(|| {
            slots.push(Slot { key, ev: 0, cl: 0 });
            (slots.len() - 1) as u32
        }) as usize
    }

    #[cold]
    #[inline(never)]
    fn find(&self, key: u64, mut s: usize) -> usize {
        loop {
            let k = self.slots[s].key;
            if k == key { return s; }
            if k == 0 { break; }
            s = (s + 1) & self.mask;
        }
        self.extra.get(&key).map_or(usize::MAX, |&i| i as usize)
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
        for round in 0..5 {
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
        use std::collections::HashMap;
        let mut hist_p = [0usize; 24];
        let mut hist_e = [0usize; 24];
        let mut hist_cp = [0usize; 24];
        let mut hist_cc = [0usize; 24];
        let (mut persons, mut sessions, mut orphan_clicks) = (0usize, 0usize, 0usize);
        let (mut buckets, mut words, mut daysp) = (0usize, 0usize, 0usize);
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
                let mut dd: Vec<u32> = ts.iter().map(|t| t / 86_400_000).collect();
                dd.dedup();
                daysp += dd.len();
            }
        }
        eprintln!("persons={persons} sessions={sessions} orphan_clicks={orphan_clicks} max_events={max_e} buckets={buckets} words={words} event_days={daysp}");
        per_bin_persons.sort_unstable();
        eprintln!("persons/bin min={} med={} max={}", per_bin_persons[0], per_bin_persons[per_bin_persons.len() / 2], per_bin_persons[per_bin_persons.len() - 1]);
        for l in 0..24 {
            if hist_p[l] + hist_cp[l] > 0 {
                eprintln!("  n<{:>8}: persons {:>8} events {:>9} | clicks: persons {:>8} clicks {:>8}", 1usize << l, hist_p[l], hist_e[l], hist_cp[l], hist_cc[l]);
            }
        }
    }
}
