//! One person's validated, linked history at a time. A session's start fixes
//! both its attribution window and its output day, even if it lasts past midnight.
//!
//! Events are never sorted: half-hour buckets keep only their earliest and
//! latest times, and neighbouring occupied buckets decide the session breaks.
//! Clicks are never sorted either: counting the session starts below a click's
//! time places it on the run of sessions whose window contains it.

use super::{cmp_keys, SlowIntern, UserRow};
use crate::group::Chain;
use crate::time::{DAY_MS, GAP_MS, WEEK_MS};

#[derive(Clone, Copy)]
pub(super) struct Hit {
    pub(super) id: u64,
    pub(super) ts: u32,
    pub(super) camp: u16,
}

#[derive(Clone, Copy, Default)]
struct Day {
    sessions: u32,
    purchasing: u32,
    events: u64,
    revenue: i64,
    top: u16,
    top_cents: i64,
}

const NO_DAY: Day = Day { sessions: 0, purchasing: 0, events: 0, revenue: 0, top: u16::MAX, top_cents: 0 };

pub(super) struct Acc {
    pub(super) rows: Vec<UserRow>,
    pub(super) stats: Vec<[i64; 4]>,
    pub(super) scratch: crate::group::Scratch,
    pub(super) phases: Phases,
    ndays: usize,
    organic: usize,
    n_cats: usize,
    // Half-hour buckets: earliest and latest time, events, then the session
    // number. Two events in one bucket are less than a gap apart, so a bucket
    // always lies inside one session; only neighbouring occupied buckets need
    // a test. One array per field keeps the read-modify-write chains of a
    // repeated bucket independent of each other.
    bmin: Box<[u32; NB]>,
    bmax: Box<[u32; NB]>,
    bcnt: Box<[u32; NB]>,
    bsess: Box<[u32; NB]>,
    occupied: Box<[u64; NB.div_ceil(64)]>,
    // Occupied buckets in time order, with room for four positions past the end.
    order: Box<[u16; NB + 4]>,
    // (bucket, category << 24 | price)
    purchases: Vec<(u32, u32)>,
    // Per session, at most one per bucket, with room for padding to whole
    // chunks of eight for the counting; `summary` holds each chunk's last start.
    starts: Vec<u32>,
    summary: Vec<u32>,
    // ends[i + 1] is the number of events in sessions 0..=i; ends[0] = 0.
    ends: Vec<u32>,
    cents: Vec<i64>,
    // Day of each session and the end of its run of same-day sessions.
    day_of: Vec<u8>,
    run_end: Vec<u32>,
    // Long path: the clicks copied out of their chain, listed by the first
    // session whose window holds them.
    flat: Vec<Hit>,
    first: Vec<u32>,
    next: Vec<u32>,
    leave: Vec<u32>,
    // Per campaign: a 16-bit epoch tag << 16 | sessions already counted as
    // assisted (long path) or its local number (short path). The table is
    // cleared whenever the tag wraps, so a stale entry never matches.
    covered: Vec<u32>,
    // Short path: per entry session the best click, tagged with the epoch;
    // per local campaign its number and assisted sessions.
    top: Vec<(u64, u64, u16)>,
    local: Vec<u16>,
    runs: Vec<u64>,
    // Short path: sessions of each day; cleared again with the rows.
    on_day: Vec<u64>,
    epoch: u64,
    days: Vec<Day>,
    // Purchase cents per (day, category), with the touched cells.
    spend: Vec<i64>,
    touched: Vec<u32>,
}

// LOW[i] has the i lowest bits set.
const LOW: [u64; 65] = {
    let mut t = [u64::MAX; 65];
    let mut i = 0;
    while i < 64 {
        t[i] = (1 << i) - 1;
        i += 1;
    }
    t
};
const GAP: u32 = GAP_MS as u32;
const WEEK: u32 = WEEK_MS as u32;
// Half-hour buckets of every 32-bit offset.
const NB: usize = (u32::MAX / GAP) as usize + 1;
const NONE: u32 = u32::MAX;

impl Acc {
    pub(super) fn new(n_camps: usize, ndays: usize, n_cats: usize) -> Self {
        let organic = n_camps - 1;
        // Offsets fit in 32 bits, so a window has at most 42 days after the
        // click week; one u64 covers its day mask.
        assert!(ndays <= 64);
        // Event offsets are below WEEK + ndays * DAY.
        let nb = (WEEK_MS as usize + ndays * DAY_MS as usize) / GAP_MS as usize + 1;
        let room = (nb + 8) & !7;
        Self {
            rows: Vec::new(),
            stats: vec![[0; 4]; n_camps * ndays],
            scratch: Default::default(),
            phases: Phases([0; 2]),
            ndays,
            organic,
            n_cats,
            bmin: Box::new([u32::MAX; NB]),
            bmax: Box::new([0; NB]),
            bcnt: Box::new([0; NB]),
            bsess: Box::new([0; NB]),
            occupied: Box::new([0; NB.div_ceil(64)]),
            order: Box::new([0; NB + 4]),
            purchases: Vec::new(),
            starts: vec![0; room],
            summary: vec![0; room / 8 + 8],
            ends: vec![0; nb + 1],
            cents: vec![0; nb],
            day_of: vec![0; nb],
            run_end: vec![0; nb],
            flat: Vec::new(),
            first: vec![NONE; nb + 1],
            next: Vec::new(),
            leave: Vec::new(),
            covered: vec![0; organic],
            top: vec![(0, 0, 0); 65],
            local: vec![0; organic],
            runs: vec![0; organic],
            on_day: vec![0; ndays],
            epoch: 0,
            days: vec![NO_DAY; ndays],
            spend: vec![0; ndays * n_cats],
            touched: Vec::new(),
        }
    }

    /// A fresh epoch for this person; its 16-bit tag is never zero, and the
    /// campaign table is cleared each time the tag wraps.
    #[inline]
    fn next_tag(&mut self) -> u32 {
        self.epoch += 1;
        if self.epoch & 0xffff == 0 {
            self.covered.fill(0);
            self.epoch += 1;
        }
        (self.epoch & 0xffff) as u32
    }

    /// Credit session `i` to a campaign and add it to its day.
    #[inline]
    fn credit(&mut self, i: usize, camp: usize, d: usize) {
        let cents = self.cents[i];
        let s = &mut self.stats[camp * self.ndays + d];
        s[0] += 1;
        s[1] += i64::from(cents != 0);
        s[2] += cents;
        let day = &mut self.days[d];
        day.sessions += 1;
        day.events += u64::from(self.ends[i + 1] - self.ends[i]);
        day.revenue += cents;
    }

    /// Up to 64 sessions, without branches on the data. The best click of
    /// each entry session is kept by selects, then carried forward; each
    /// campaign's assisted sessions are a bit mask. Returns the day mask.
    #[inline(never)]
    fn attribute_short(&mut self, ns: usize, clicks: Chain<Hit>, slow: &SlowIntern) -> u64 {
        // Pad the starts to whole groups of eight for the counting.
        let p = (ns + 7) & !7;
        self.starts[ns..p].fill(u32::MAX);
        let tag = self.next_tag();
        let epoch = self.epoch;
        let mut nl = 0usize;
        let starts = &self.starts[..p];
        let (top, covered, local, runs) = (&mut self.top[..], &mut self.covered[..], &mut self.local[..], &mut self.runs[..]);
        clicks.for_each(|h| {
            // Session i holds the click when start - WEEK <= ts <= start:
            // a = #(start < ts) sessions come before, b = #(start <= ts + WEEK).
            let (a, b) = below(starts, h.ts, h.ts.saturating_add(WEEK + 1));
            // Per entry session the best click: epoch << 32 | 1 + time, its ID
            // and campaign. An entry left by an earlier person loses to any
            // click of this one, so nothing needs clearing. a <= b <= ns <= 64.
            // SAFETY: `top` has 65 entries.
            let top = unsafe { top.get_unchecked_mut(a) };
            let (t, id, camp) = *top;
            let t2 = epoch << 32 | u64::from(h.ts) + 1;
            // Computed before the time test, which keeps this free of branches.
            let smaller = id_lt(h.id, id, slow);
            let better = t2 > t || (t2 == t && smaller);
            *top = (if better { t2 } else { t }, if better { h.id } else { id }, if better { h.camp } else { camp });
            let k = usize::from(h.camp);
            let c = covered[k];
            let new = c >> 16 != tag;
            let x = if new { nl } else { (c & 0xffff) as usize };
            nl += usize::from(new);
            covered[k] = tag << 16 | x as u32;
            // A new local number starts from an empty mask.
            let keep = u64::from(new).wrapping_sub(1);
            local[x] = h.camp;
            runs[x] = (LOW[b] & !LOW[a]) | (runs[x] & keep);
        });
        // Later entry sessions hold strictly later clicks, so a filled entry
        // always replaces the carried one. The same pass records each
        // session's day and each day's sessions for the assists below.
        let (mut t, mut camp) = (0u32, 0u16);
        let mut mask = 0u64;
        for i in 0..ns {
            let (t2, _, camp2) = self.top[i];
            let take = t2 >> 32 == epoch;
            (t, camp) = (if take { t2 as u32 } else { t }, if take { camp2 } else { camp });
            let d = day_index(self.starts[i]);
            self.day_of[i] = d as u8;
            self.on_day[d] |= 1 << i;
            mask |= 1 << d;
            let c = if t > self.starts[i] - WEEK { usize::from(camp) } else { self.organic };
            self.credit(i, c, d);
        }
        for x in 0..nl {
            let (k, mut r) = (usize::from(self.local[x]), self.runs[x]);
            // Same-day sessions are adjacent in the mask: each round takes
            // the run's sessions on the day of its lowest remaining one.
            while r != 0 {
                let d = usize::from(self.day_of[r.trailing_zeros() as usize]);
                let on = self.on_day[d];
                self.stats[k * self.ndays + d][3] += i64::from((r & on).count_ones());
                r &= !on;
            }
        }
        mask
    }

    /// Session days and same-day runs, plus the padding and chunk summary
    /// for `below2`. Returns the padded length.
    #[inline(never)]
    fn prepare(&mut self, ns: usize) -> usize {
        // At least one padding entry keeps the last chunk's summary at MAX.
        let p = (ns + 8) & !7;
        for i in 0..ns {
            self.day_of[i] = day_index(self.starts[i]) as u8;
        }
        self.starts[ns..p].fill(u32::MAX);
        let q = (p / 8 + 7) & !7;
        for j in 0..p / 8 {
            self.summary[j] = self.starts[8 * j + 7];
        }
        self.summary[p / 8..q].fill(u32::MAX);
        let mut end = ns as u32;
        for i in (0..ns).rev() {
            self.run_end[i] = end;
            if i == 0 || self.day_of[i - 1] != self.day_of[i] {
                end = i as u32;
            }
        }
        p
    }

    /// Credit every session from its latest click within a week and count
    /// each campaign's assists as the union of its clicks' session runs.
    /// Returns the day mask.
    #[inline(never)]
    fn attribute(&mut self, ns: usize, clicks: Chain<Hit>, slow: &SlowIntern) -> u64 {
        let p = self.prepare(ns);
        let q = (p / 8 + 7) & !7;
        let mut flat = std::mem::take(&mut self.flat);
        flat.clear();
        clicks.for_each(|h| flat.push(*h));
        let clicks = &flat[..];
        // Click j is inside the windows of sessions [enter, leave):
        // enter = #(start < ts), leave = #(start <= ts + WEEK).
        self.next.resize(clicks.len(), NONE);
        self.leave.resize(clicks.len(), 0);
        self.first[..=ns].fill(NONE);
        for (j, h) in clicks.iter().enumerate() {
            let (a, b) = below2(&self.starts[..p], &self.summary[..q], h.ts, h.ts.saturating_add(WEEK + 1));
            self.leave[j] = b as u32;
            self.next[j] = self.first[a];
            self.first[a] = j as u32;
        }
        let tag = self.next_tag();
        // Latest click at or before the current start; ties go to the smaller ID.
        let (mut best_ts, mut best_id, mut best_camp) = (-1i64, 0u64, 0usize);
        let mut mask = 0u64;
        for i in 0..ns {
            let mut j = self.first[i];
            while j != NONE {
                let h = clicks[j as usize];
                let ts = i64::from(h.ts);
                if ts > best_ts || (ts == best_ts && id_lt(h.id, best_id, slow)) {
                    (best_ts, best_id, best_camp) = (ts, h.id, usize::from(h.camp));
                }
                let k = usize::from(h.camp);
                let c = self.covered[k];
                let from = if c >> 16 == tag { (c & 0xffff).max(i as u32) } else { i as u32 };
                let to = self.leave[j as usize];
                if to > from {
                    // Sessions number at most the buckets, far below 2^16.
                    self.covered[k] = tag << 16 | to;
                    let mut x = from as usize;
                    while x < to as usize {
                        let y = (self.run_end[x] as usize).min(to as usize);
                        self.stats[k * self.ndays + usize::from(self.day_of[x])][3] += (y - x) as i64;
                        x = y;
                    }
                }
                j = self.next[j as usize];
            }
            let d = usize::from(self.day_of[i]);
            let camp = if best_ts + i64::from(WEEK) >= i64::from(self.starts[i]) { best_camp } else { self.organic };
            self.credit(i, camp, d);
            mask |= 1 << d;
        }
        self.flat = flat;
        mask
    }

    /// Sessions of a few events: a fixed comparator network sorts them, and
    /// one sweep finds the breaks. Leaves starts, ends and purchases
    /// (session, pay); returns the session count.
    #[inline(never)]
    fn sessions_sorted(&mut self, v: &mut [u64; 16], n: usize) -> usize {
        if n <= 8 {
            sort8(v);
        } else {
            sort16(v);
        }
        let (starts, ends) = (&mut self.starts[..], &mut self.ends[..]);
        let purchases = &mut self.purchases;
        purchases.clear();
        ends[0] = 0;
        // The first offset is at least a week, so it opens a session. Every
        // event writes the start of a would-be next session, which the real
        // next start overwrites.
        let mut ns = 0;
        let mut prev = 0u32;
        for (i, &e) in v[..n].iter().enumerate() {
            let ts = timestamp(e);
            let new = ts - prev > GAP;
            starts[ns] = ts;
            ns += usize::from(new);
            ends[ns] = i as u32 + 1;
            prev = ts;
            // Only the price marks a purchase: the contract leaves the
            // category bits of a view or cart unspecified.
            let pay = e as u32;
            if pay & 0x00ff_ffff != 0 {
                purchases.push((ns as u32 - 1, pay));
            }
        }
        ns
    }

    /// Sessions of many events through the bucket summaries. Leaves starts,
    /// ends and purchases (session, pay); returns the session count.
    #[inline(never)]
    fn sessions_bucketed(&mut self, events: Chain<u64>) -> usize {
        // Bucket the events instead of sorting them. Locals keep the slices
        // in registers; a 32-bit time divided by the gap indexes the fixed
        // tables without bounds checks.
        // Buckets of 64 form words; `words` marks the words with events so
        // the sweep below skips the empty ones (a 32-bit offset has 38 words).
        let mut words = 0u64;
        let purchases = &mut self.purchases;
        purchases.clear();
        {
            let (bmin, bmax, bcnt) = (&mut *self.bmin, &mut *self.bmax, &mut *self.bcnt);
            let occupied = &mut *self.occupied;
            events.for_each_chunk(|chunk| {
                for &e in chunk {
                    let ts = timestamp(e);
                    let b = (ts / GAP) as usize;
                    bmin[b] = bmin[b].min(ts);
                    bmax[b] = bmax[b].max(ts);
                    bcnt[b] += 1;
                    occupied[b / 64] |= 1 << (b % 64);
                    words |= 1 << (b / 64);
                    // Only the price marks a purchase (see `sessions_sorted`).
                    let pay = e as u32;
                    if pay & 0x00ff_ffff != 0 {
                        purchases.push((b as u32, pay));
                    }
                }
            });
        }
        // Occupied buckets in time order: four positions per word are written
        // unconditionally and the end advances by the word's population, so
        // only words with more than four buckets branch.
        let order = &mut *self.order;
        let mut nb = 0;
        while words != 0 {
            let w = words.trailing_zeros() as usize;
            words &= words - 1;
            let mut m = std::mem::take(&mut self.occupied[w]);
            let n = m.count_ones() as usize;
            let base = (w * 64) as u16;
            let out = &mut order[nb..nb + 4];
            for x in out {
                *x = base + m.trailing_zeros() as u16;
                m &= m.wrapping_sub(1);
            }
            if n > 4 {
                let mut j = nb + 4;
                while m != 0 {
                    order[j] = base + m.trailing_zeros() as u16;
                    m &= m - 1;
                    j += 1;
                }
            }
            nb += n;
        }
        // Event offsets are at least a week, so the first bucket always
        // opens a session. Every bucket writes the start of a would-be next
        // session, which the real next start overwrites.
        let mut ns = 0;
        let mut prev = 0u32;
        let mut cum = 0u32;
        {
            let (bmin, bmax, bcnt, bsess) = (&mut *self.bmin, &mut *self.bmax, &mut *self.bcnt, &mut *self.bsess);
            let (starts, ends) = (&mut self.starts[..], &mut self.ends[..]);
            ends[0] = 0;
            for &b in &order[..nb] {
                let b = usize::from(b);
                let (min, max, n) = (bmin[b], bmax[b], bcnt[b]);
                let new = min - prev > GAP;
                starts[ns] = min;
                ns += usize::from(new);
                cum += n;
                ends[ns] = cum;
                bsess[b] = (ns - 1) as u32;
                bmin[b] = u32::MAX;
                bmax[b] = 0;
                bcnt[b] = 0;
                prev = max;
            }
        }
        for p in self.purchases.iter_mut() {
            p.0 = self.bsess[p.0 as usize];
        }
        ns
    }

    /// Session cents, and each day's top category from the purchases.
    #[inline(never)]
    fn apply_purchases(&mut self, ns: usize, cats: &[Vec<u8>]) {
        self.cents[..ns].fill(0);
        for &(i, pay) in &self.purchases {
            let i = i as usize;
            let price = i64::from(pay & 0x00ff_ffff);
            self.cents[i] += price;
            let cell = day_index(self.starts[i]) * self.n_cats + (pay >> 24) as usize;
            if self.spend[cell] == 0 {
                self.touched.push(cell as u32);
            }
            self.spend[cell] += price;
        }
        for &cell in &self.touched {
            let cell = cell as usize;
            let (d, c) = (cell / self.n_cats, cell % self.n_cats);
            let cents = std::mem::take(&mut self.spend[cell]);
            let day = &mut self.days[d];
            if day.top == u16::MAX || cents > day.top_cents || (cents == day.top_cents && cats[c] < cats[usize::from(day.top)]) {
                day.top = c as u16;
                day.top_cents = cents;
            }
        }
        self.touched.clear();
    }

    /// A person without clicks: every session is organic. Returns the day mask.
    #[inline(never)]
    fn organic(&mut self, ns: usize) -> u64 {
        let mut mask = 0u64;
        for i in 0..ns {
            let d = day_index(self.starts[i]);
            let cents = self.cents[i];
            let day = &mut self.days[d];
            day.sessions += 1;
            day.purchasing += u32::from(cents != 0);
            day.events += u64::from(self.ends[i + 1] - self.ends[i]);
            day.revenue += cents;
            mask |= 1 << d;
        }
        let mut m = mask;
        while m != 0 {
            let d = m.trailing_zeros() as usize;
            m &= m - 1;
            let day = self.days[d];
            let s = &mut self.stats[self.organic * self.ndays + d];
            s[0] += i64::from(day.sessions);
            s[1] += i64::from(day.purchasing);
            s[2] += day.revenue;
        }
        mask
    }

    #[inline(never)]
    pub(super) fn person(
        &mut self,
        key: u64,
        events: Chain<u64>,
        clicks: Chain<Hit>,
        cats: &[Vec<u8>],
        slow: &SlowIntern,
        first_day: i32,
    ) {
        if events.is_empty() {
            return;
        }
        // Up to two chunks hold at most fourteen events: those are sorted.
        let (last, prev) = events.last();
        let ns = if prev == 0 {
            let mut v = [u64::MAX; 16];
            v[..last.len()].copy_from_slice(last);
            self.sessions_sorted(&mut v, last.len())
        } else {
            let (full, before) = events.full(prev);
            if before == 0 {
                let mut v = [u64::MAX; 16];
                v[..7].copy_from_slice(full);
                v[7..7 + last.len()].copy_from_slice(last);
                self.sessions_sorted(&mut v, 7 + last.len())
            } else {
                self.sessions_bucketed(events)
            }
        };
        self.apply_purchases(ns, cats);
        let mut mask = if clicks.is_empty() {
            self.organic(ns)
        } else if ns <= 64 {
            self.attribute_short(ns, clicks, slow)
        } else {
            self.attribute(ns, clicks, slow)
        };
        while mask != 0 {
            let d = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            self.on_day[d] = 0;
            let day = std::mem::replace(&mut self.days[d], NO_DAY);
            self.rows.push(UserRow {
                key,
                day: first_day + d as i32,
                sessions: i64::from(day.sessions),
                events: day.events as i64,
                revenue: day.revenue,
                top: day.top,
            });
        }
    }
}

/// Nanoseconds a worker spent chaining rows onto persons and summarizing
/// them. The stage report in `stream` has no view inside the bins; each
/// worker's split is reported when the report is asked for.
pub(super) struct Phases(pub(super) [u64; 2]);

impl Drop for Phases {
    fn drop(&mut self) {
        if self.0 != [0; 2] && std::env::var_os("STAGES").is_some() {
            let [a, b] = self.0.map(|n| n / 1_000_000);
            eprintln!("group phases ms chain={a} person={b}");
        }
    }
}

/// Bytewise ID order; packed IDs compare as integers. Zero stands for "no
/// click yet", where the answer is unused.
#[inline(always)]
fn id_lt(a: u64, b: u64, slow: &SlowIntern) -> bool {
    if (a | b) >> 63 == 0 { a < b } else { slow_lt(a, b, slow) }
}

#[cold]
#[inline(never)]
fn slow_lt(a: u64, b: u64, slow: &SlowIntern) -> bool {
    a != 0 && b != 0 && cmp_keys(a, b, slow).is_lt()
}

/// How many entries of a sorted slice, whose length is a multiple of eight,
/// are below x and below y: eight lanes at a time, no data-dependent steps.
/// The three versions compute the same counts.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
fn below(a: &[u32], x: u32, y: u32) -> (usize, usize) {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 is a compile-time target feature here; every chunk holds
    // eight u32 values, and unaligned loads need no alignment.
    unsafe {
        // Unsigned order is signed order with the top bit flipped.
        let flip = _mm256_set1_epi32(i32::MIN);
        let vx = _mm256_xor_si256(_mm256_set1_epi32(x as i32), flip);
        let vy = _mm256_xor_si256(_mm256_set1_epi32(y as i32), flip);
        let (mut cx, mut cy) = (0u32, 0u32);
        for c in a.chunks_exact(8) {
            let v = _mm256_xor_si256(_mm256_loadu_si256(c.as_ptr().cast()), flip);
            cx += (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vx, v))) as u32).count_ones();
            cy += (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vy, v))) as u32).count_ones();
        }
        (cx as usize, cy as usize)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn below(a: &[u32], x: u32, y: u32) -> (usize, usize) {
    use std::arch::aarch64::*;
    // SAFETY: NEON is always present on aarch64; every chunk holds eight u32
    // values, loaded from in-bounds pointers.
    unsafe {
        let vx = vdupq_n_u32(x);
        let vy = vdupq_n_u32(y);
        let (mut cx, mut cy) = (vdupq_n_u32(0), vdupq_n_u32(0));
        for c in a.chunks_exact(8) {
            let lo = vld1q_u32(c.as_ptr());
            let hi = vld1q_u32(c.as_ptr().add(4));
            // A true lane is all ones, so subtracting it counts one.
            cx = vsubq_u32(vsubq_u32(cx, vcgtq_u32(vx, lo)), vcgtq_u32(vx, hi));
            cy = vsubq_u32(vsubq_u32(cy, vcgtq_u32(vy, lo)), vcgtq_u32(vy, hi));
        }
        (vaddvq_u32(cx) as usize, vaddvq_u32(cy) as usize)
    }
}

#[cfg(not(any(all(target_arch = "x86_64", target_feature = "avx2"), target_arch = "aarch64")))]
#[inline(always)]
fn below(a: &[u32], x: u32, y: u32) -> (usize, usize) {
    let (mut cx, mut cy) = (0usize, 0usize);
    for &s in a {
        cx += usize::from(s < x);
        cy += usize::from(s < y);
    }
    (cx, cy)
}

/// `below` over a long slice in two steps: `summary` holds the last entry
/// of each chunk of eight (padded with u32::MAX, like the slice), so the
/// chunks wholly below a value are counted first, then that value's chunk.
#[inline(always)]
fn below2(a: &[u32], summary: &[u32], x: u32, y: u32) -> (usize, usize) {
    let (cx, cy) = below(summary, x, y);
    // The padding chunk ends in u32::MAX, so cx and cy index real chunks.
    let (ax, _) = below(&a[8 * cx..8 * cx + 8], x, x);
    let (_, by) = below(&a[8 * cy..8 * cy + 8], y, y);
    (8 * cx + ax, 8 * cy + by)
}

/// Compare-exchange the listed pairs in turn, each a min and a max.
macro_rules! network {
    ($v:ident; $(($i:literal, $j:literal)),* $(,)?) => {
        $( { let (a, b) = ($v[$i], $v[$j]); $v[$i] = a.min(b); $v[$j] = a.max(b); } )*
    };
}

/// Sort the first eight entries: 19 comparators in 6 layers.
#[inline(always)]
fn sort8(v: &mut [u64; 16]) {
    network!(v;
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (1, 2), (5, 6),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (2, 4), (3, 5),
        (1, 2), (3, 4), (5, 6),
    );
}

/// Sort all sixteen entries: 60 comparators in 10 layers.
#[inline(always)]
fn sort16(v: &mut [u64; 16]) {
    network!(v;
        (0, 13), (1, 12), (2, 15), (3, 14), (4, 8), (5, 6), (7, 11), (9, 10),
        (0, 5), (1, 7), (2, 9), (3, 4), (6, 13), (8, 14), (10, 15), (11, 12),
        (0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 10), (5, 11), (6, 7), (8, 9), (12, 14), (13, 15),
        (1, 2), (3, 12), (4, 6), (5, 7), (8, 10), (9, 11), (13, 14),
        (1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14),
        (2, 4), (3, 6), (9, 12), (11, 13),
        (3, 5), (6, 8), (7, 9), (10, 12),
        (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
        (6, 7), (8, 9),
    );
}

#[inline]
fn timestamp(event: u64) -> u32 {
    (event >> 32) as u32
}

#[inline]
fn day_index(start: u32) -> usize {
    ((start - WEEK) / DAY_MS as u32) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u32 = DAY_MS as u32;

    fn ev(ts: u32, cat: u8, cents: u32) -> u64 {
        (u64::from(ts) << 32) | (u64::from(cat) << 24) | u64::from(cents)
    }

    fn row(r: &UserRow) -> (u64, i32, i64, i64, i64, u16) {
        (r.key, r.day, r.sessions, r.events, r.revenue, r.top)
    }

    /// `person` on plain slices.
    #[inline(never)]
    fn person(acc: &mut Acc, key: u64, events: &[u64], clicks: &[Hit], cats: &[Vec<u8>], slow: &SlowIntern, first_day: i32) {
        let (mut ea, mut ca) = (Vec::new(), Vec::new());
        let ev = Chain::build(events, &mut ea);
        let cl = Chain::build(clicks, &mut ca);
        acc.person(key, ev, cl, cats, slow, first_day);
    }

    #[test]
    fn session_gaps_midnight_and_category_totals() {
        let slow = SlowIntern::new();
        let cats = vec![b"zulu".to_vec(), b"unknown".to_vec()];
        let mut acc = Acc::new(1, 16, cats.len());
        let start = WEEK + DAY - 1;
        let after_midnight = start + GAP;
        let next = after_midnight + GAP + 1;
        let events = [
            ev(next + GAP + 1, 1, 0),
            ev(WEEK + 15 * DAY, 0, 0),
            ev(after_midnight, 1, 35),
            ev(next, 0, 110),
            ev(after_midnight, 0, 0),
            ev(start, 0, 75),
            ev(after_midnight, 1, 40),
        ];
        person(&mut acc, 41, &events, &[], &cats, &slow, -2);
        assert_eq!(
            acc.rows.iter().map(row).collect::<Vec<_>>(),
            [
                (41, -2, 1, 4, 150, 1), // "unknown" wins a 75:75 tie.
                (41, -1, 2, 2, 110, 0),
                (41, 13, 1, 1, 0, u16::MAX),
            ]
        );
        let mut expected = vec![[0; 4]; 16];
        expected[0] = [1, 1, 150, 0];
        expected[1] = [2, 1, 110, 0];
        expected[15] = [1, 0, 0, 0];
        assert_eq!(acc.stats, expected);
    }

    // The contract fixes only the price of a view or cart at zero; on both
    // the sorted and the bucketed path a category alone is no purchase.
    #[test]
    fn view_categories_are_not_purchases() {
        let slow = SlowIntern::new();
        let cats = vec![b"b".to_vec(), b"a".to_vec()];
        let mut acc = Acc::new(1, 1, cats.len());
        person(&mut acc, 5, &[ev(WEEK, 1, 0), ev(WEEK + 1, 0, 0)], &[], &cats, &slow, 0);
        assert_eq!(acc.rows.iter().map(row).collect::<Vec<_>>(), [(5, 0, 1, 2, 0, u16::MAX)]);
        let many: Vec<u64> = (0..20u32).map(|i| ev(WEEK + i * (GAP + 1), 1, 0)).collect();
        person(&mut acc, 6, &many, &[], &cats, &slow, 0);
        assert_eq!(row(&acc.rows[1]), (6, 0, 20, 20, 0, u16::MAX));
        assert_eq!(acc.stats, [[21, 0, 0, 0]]);
    }

    // Every history size up to the bucketed path, with the same answer as a
    // plain sort of the events gives.
    #[test]
    fn sorted_and_bucketed_sessions_agree_at_every_size() {
        let slow = SlowIntern::new();
        let cats = vec![b"a".to_vec(), b"b".to_vec()];
        let mut seed = 0x1234_5678u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for n in 1..40usize {
            for _ in 0..20 {
                let mut t = WEEK + rnd(u64::from(2 * DAY)) as u32;
                let mut events = Vec::new();
                for _ in 0..n {
                    events.push(ev(t, rnd(2) as u8, if rnd(3) == 0 { 1 + rnd(50) as u32 } else { 0 }));
                    t += rnd(u64::from(2 * GAP + 2)) as u32;
                }
                // Expected rows by sorting.
                let mut sorted: Vec<u32> = events.iter().map(|&e| timestamp(e)).collect();
                sorted.sort_unstable();
                let mut want = std::collections::BTreeMap::new();
                let mut start = sorted[0];
                let mut prev = sorted[0];
                let mut sess: Vec<(u32, u32)> = Vec::new(); // (start, last ts)
                for &ts in &sorted[1..] {
                    if ts - prev > GAP {
                        sess.push((start, prev));
                        start = ts;
                    }
                    prev = ts;
                }
                sess.push((start, prev));
                for &(s, e) in &sess {
                    let d = day_index(s);
                    let entry = want.entry(d).or_insert((0i64, 0i64, 0i64));
                    entry.0 += 1;
                    for &x in &events {
                        let ts = timestamp(x);
                        if ts >= s && ts <= e {
                            entry.1 += 1;
                            entry.2 += i64::from(x as u32 & 0x00ff_ffff);
                        }
                    }
                }
                let mut acc = Acc::new(1, 4, cats.len());
                person(&mut acc, 9, &events, &[], &cats, &slow, 0);
                let got: Vec<_> = acc.rows.iter().map(|r| (r.day as usize, (r.sessions, r.events, r.revenue))).collect();
                let want: Vec<_> = want.into_iter().collect();
                assert_eq!(got, want, "n={n}");
            }
        }
    }

    #[test]
    fn inclusive_click_window_ties_and_duplicate_assists() {
        let mut slow = SlowIntern::new();
        let mut hit = |ts, id: &[u8], camp| Hit {
            id: crate::try_pack(id).unwrap_or_else(|| slow.intern(id)),
            ts,
            camp,
        };
        let a = WEEK + 1;
        let b = a + GAP + 1;
        let c = a + 1 + WEEK;
        let d = c + GAP + 1;
        let dup = hit(a, b"k2", 2);
        let clicks = [
            hit(d + 1, b"kfuture", 0),
            dup,
            hit(1, b"katweek", 1),
            hit(a, b"!slow-earliest-id", 3),
            hit(0, b"kexpired", 0),
            hit(a + 1, b"kafter", 4),
            dup,
            hit(a - 1, b"kother", 2),
        ];
        let events = [ev(d, 0, 42), ev(b, 0, 20), ev(c, 0, 0), ev(a, 0, 10)];
        let cats = vec![b"book".to_vec()];
        let mut acc = Acc::new(6, 8, cats.len());
        person(&mut acc, 1, &events, &clicks, &cats, &slow, 0);

        let mut expected = vec![[0; 4]; 6 * 8];
        expected[8] = [0, 0, 0, 1]; // Exactly seven days before A, too old for B.
        expected[2 * 8] = [0, 0, 0, 2];
        expected[3 * 8] = [1, 1, 10, 2]; // Slow ID sorts before the packed ID.
        expected[4 * 8] = [1, 1, 20, 1]; // One millisecond after A.
        expected[4 * 8 + 7] = [1, 0, 0, 1]; // Same click credits C, exactly a week later.
        expected[5 * 8 + 7] = [1, 1, 42, 0];
        assert_eq!(acc.stats, expected);
    }

    #[test]
    fn reuse_scratch_across_people_and_sessions() {
        let mut slow = SlowIntern::new();
        let cats = vec![b"z".to_vec(), b"a".to_vec()];
        let mut acc = Acc::new(3, 1, cats.len());
        let first = Hit { ts: 0, id: slow.intern(b"first"), camp: 0 };
        let later = Hit { ts: WEEK, id: slow.intern(b"later"), camp: 1 };
        let refresh = Hit { ts: WEEK + GAP + 2, id: slow.intern(b"refresh"), camp: 1 };
        let events = [
            ev(WEEK + 2 * (GAP + 1), 0, 16),
            ev(WEEK, 1, 20),
            ev(WEEK + GAP + 1, 0, 4),
        ];
        person(&mut acc, 1, &events, &[first, later, refresh], &cats, &slow, 7);
        // Camp 0 expires, camp 1 moves to its slot, then gets another click.
        // The two later sessions together tie category 1 on daily revenue.
        assert_eq!(row(&acc.rows[0]), (1, 7, 3, 3, 40, 1));
        person(&mut acc, 2, &[ev(WEEK, 0, 2)], &[first], &cats, &slow, 7);
        person(&mut acc, 3, &[ev(WEEK, 0, 0)], &[], &cats, &slow, 7);
        person(&mut acc, 4, &[], &[later], &cats, &slow, 7);
        assert_eq!(acc.rows.len(), 3);
        assert_eq!(row(&acc.rows[1]), (2, 7, 1, 1, 2, 0));
        assert_eq!(row(&acc.rows[2]), (3, 7, 1, 1, 0, u16::MAX));
        assert_eq!(acc.stats, [[1, 1, 2, 2], [3, 3, 40, 3], [1, 0, 0, 0]]);
    }

    // The campaign table's epoch tag wraps after 65535 persons with clicks.
    // A campaign last touched under the same tag one wrap earlier must not
    // look current: persons right at the wrap, on the short and the long
    // path, must get what fresh accumulators give them.
    #[test]
    fn campaign_table_survives_the_epoch_wrap() {
        let slow = SlowIntern::new();
        let cats = vec![b"a".to_vec()];
        let short_ev = [ev(WEEK + 5, 0, 7), ev(WEEK + GAP + 6, 0, 0)];
        let short_cl = |c: [u16; 2]| [Hit { ts: WEEK, id: 1, camp: c[0] }, Hit { ts: WEEK + 1, id: 2, camp: c[1] }, Hit { ts: 3, id: 3, camp: c[0] }];
        let long_ev: Vec<u64> = (0..70u32).map(|i| ev(WEEK + i * (GAP + 1), 0, u32::from(i % 3 == 0))).collect();
        let long_cl = |c: [u16; 2]| -> Vec<Hit> {
            (0..40u32).map(|j| Hit { ts: WEEK + j * 3 * (GAP + 1) / 2, id: 10 + u64::from(j), camp: c[(j % 2) as usize] }).collect()
        };
        let (a, b) = ([2u16, 5], [0u16, 1]);
        let one = |evs: &[u64], cls: &[Hit]| {
            let mut acc = Acc::new(7, 3, cats.len());
            person(&mut acc, 1, evs, cls, &cats, &slow, 0);
            acc.stats
        };
        let (sa, sb, la) = (one(&short_ev, &short_cl(a)), one(&short_ev, &short_cl(b)), one(&long_ev, &long_cl(a)));
        assert!(la.iter().any(|c| c[3] > 1), "the long path must assist");
        let mut acc = Acc::new(7, 3, cats.len());
        let (mut na, mut nb) = (0i64, 0i64);
        // Campaigns 2 and 5 are touched under tag 1, then again by the first
        // person after each wrap, whose tag is 1 as well.
        for i in 0..=2 * 65535 {
            let wrap = i == 0 || i == 65535;
            if i == 2 * 65535 {
                person(&mut acc, i + 1, &long_ev, &long_cl(a), &cats, &slow, 0);
            } else {
                let c = if wrap { a } else { b };
                person(&mut acc, i + 1, &short_ev, &short_cl(c), &cats, &slow, 0);
                if wrap { na += 1 } else { nb += 1 }
            }
        }
        let expected: Vec<[i64; 4]> =
            (0..sa.len()).map(|c| std::array::from_fn(|f| sa[c][f] * na + sb[c][f] * nb + la[c][f])).collect();
        assert_eq!(acc.stats, expected);
    }

    // By the 0-1 principle a network that sorts every 0/1 input sorts all.
    #[test]
    fn comparator_networks_sort() {
        for x in 0..1u32 << 16 {
            let mut v = [0u64; 16];
            for i in 0..16 {
                v[i] = u64::from(x >> i & 1);
            }
            let mut w = v;
            sort16(&mut w);
            assert!(w.windows(2).all(|p| p[0] <= p[1]), "{x:016b}");
            if x < 1 << 8 {
                v[8..].fill(u64::MAX);
                sort8(&mut v);
                assert!(v.windows(2).all(|p| p[0] <= p[1]), "{x:08b}");
            }
        }
    }

    // Many sessions on many days with clicks in every relation to the
    // windows: the short and long paths must agree with each other.
    #[test]
    fn short_and_long_paths_agree() {
        let slow = SlowIntern::new();
        let cats = vec![b"a".to_vec(), b"b".to_vec()];
        let mut seed = 0x9e37_79b9u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let span = 14 * DAY;
        let limit = WEEK + span - 2 * GAP;
        for round in 0..60 {
            let ns = 1 + rnd(if round % 2 == 0 { 64 } else { 200 }) as usize;
            let mut events = Vec::new();
            let mut t = WEEK + rnd(u64::from(DAY)) as u32;
            'outer: for _ in 0..ns {
                for _ in 0..1 + rnd(4) {
                    if t >= limit {
                        break 'outer;
                    }
                    events.push(ev(t, rnd(2) as u8, if rnd(5) == 0 { 10 + rnd(100) as u32 } else { 0 }));
                    t += rnd(u64::from(GAP)) as u32;
                }
                t += GAP + 1 + rnd(u64::from(span) / (ns as u64 * 2)) as u32;
            }
            let clicks: Vec<Hit> = (0..rnd(300))
                .map(|j| Hit { ts: rnd(u64::from(WEEK + span)) as u32, id: 1 + j, camp: rnd(9) as u16 })
                .collect();
            let mut acc_a = Acc::new(10, 18, cats.len());
            person(&mut acc_a, 1, &events, &clicks, &cats, &slow, 0);
            // The same history under the long path: 65 isolated sessions are
            // added after the 14 days, which only touch later cells.
            let mut acc_b = Acc::new(10, 18, cats.len());
            for i in 0..65u32 {
                events.push(ev(WEEK + span + i * (GAP + 1) * 2, 0, 0));
            }
            person(&mut acc_b, 1, &events, &clicks, &cats, &slow, 0);
            let strip = |acc: &Acc| -> Vec<[i64; 4]> {
                acc.stats.chunks(18).flat_map(|c| c[..14].iter().copied()).collect()
            };
            assert_eq!(strip(&acc_a), strip(&acc_b), "round {round}: ns={ns} clicks={}", clicks.len());
            let ra: Vec<_> = acc_a.rows.iter().map(row).collect();
            let rb: Vec<_> = acc_b.rows.iter().map(row).filter(|r| r.1 < 14).collect();
            assert_eq!(ra, rb);
        }
    }
}
