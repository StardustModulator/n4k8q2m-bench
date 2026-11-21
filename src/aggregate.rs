//! One person's validated, linked history at a time. A session's start fixes
//! both its attribution window and its output day, even if it lasts past midnight.
//!
//! Events are never sorted: half-hour buckets keep only their earliest and
//! latest times, and neighbouring occupied buckets decide the session breaks.
//! Clicks are never sorted either: counting the session starts below a click's
//! time places it on the run of sessions whose window contains it.

use super::{cmp_keys, SlowIntern, UserRow};
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
    // Per session, at most one per bucket. Starts and lowers (start - WEEK)
    // have room for padding to a power of two for the searches.
    starts: Vec<u32>,
    lowers: Vec<u32>,
    // ends[i + 1] is the number of events in sessions 0..=i; ends[0] = 0.
    ends: Vec<u32>,
    cents: Vec<i64>,
    // Day of each session and the end of its run of same-day sessions.
    day_of: Vec<u8>,
    run_end: Vec<u32>,
    // Long path: clicks listed by the first session whose window holds them.
    first: Vec<u32>,
    next: Vec<u32>,
    leave: Vec<u32>,
    // Per campaign: epoch << 32 | sessions already counted as assisted (long
    // path) or its local number (short path).
    covered: Vec<u64>,
    // Short path: per entry session (1 + time, ID, campaign) of the best
    // click; per local campaign its number and assisted sessions.
    top: Vec<(u32, u64, u16)>,
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
        let room = (nb + 1).next_power_of_two();
        Self {
            rows: Vec::new(),
            stats: vec![[0; 4]; n_camps * ndays],
            scratch: Default::default(),
            phases: Phases([0; 3]),
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
            lowers: vec![0; room],
            ends: vec![0; nb + 1],
            cents: vec![0; nb],
            day_of: vec![0; nb],
            run_end: vec![0; nb],
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
    fn attribute_short(&mut self, ns: usize, clicks: &[Hit], slow: &SlowIntern) -> u64 {
        // Pad the starts to whole groups of eight for the counting.
        let p = (ns + 7) & !7;
        self.starts[ns..p].fill(u32::MAX);
        let mut mask = 0u64;
        for i in 0..ns {
            let d = day_index(self.starts[i]);
            self.day_of[i] = d as u8;
            self.on_day[d] |= 1 << i;
            mask |= 1 << d;
        }
        // Per entry session: one plus the best time, its ID and campaign.
        self.top[..=ns].fill((0, 0, 0));
        self.epoch += 1;
        let epoch = self.epoch;
        let mut nl = 0usize;
        let starts = &self.starts[..p];
        for h in clicks {
            // Session i holds the click when start - WEEK <= ts <= start:
            // a = #(start < ts) sessions come before, b = #(start <= ts + WEEK).
            let (a, b) = below(starts, h.ts, h.ts.saturating_add(WEEK + 1));
            // a <= b <= ns <= 64.
            // SAFETY: `top` has 65 entries.
            let top = unsafe { self.top.get_unchecked_mut(a) };
            let (t, id, camp) = *top;
            // Computed before the time test, which keeps this free of branches.
            let smaller = id_lt(h.id, id, slow);
            let t2 = h.ts + 1;
            let better = t2 > t || (t2 == t && smaller);
            *top = (if better { t2 } else { t }, if better { h.id } else { id }, if better { h.camp } else { camp });
            let k = usize::from(h.camp);
            let c = self.covered[k];
            let new = c >> 32 != epoch;
            let x = if new { nl } else { c as u32 as usize };
            nl += usize::from(new);
            self.covered[k] = epoch << 32 | x as u64;
            // A new local number starts from an empty mask.
            let keep = u64::from(new).wrapping_sub(1);
            self.local[x] = h.camp;
            self.runs[x] = (LOW[b] & !LOW[a]) | (self.runs[x] & keep);
        }
        // Later entry sessions hold strictly later clicks, so a filled entry
        // always replaces the carried one.
        let (mut t, mut camp) = (0u32, 0u16);
        for i in 0..ns {
            let (t2, _, camp2) = self.top[i];
            let take = t2 != 0;
            (t, camp) = (if take { t2 } else { t }, if take { camp2 } else { camp });
            let d = usize::from(self.day_of[i]);
            let c = if t > self.starts[i] - WEEK { usize::from(camp) } else { self.organic };
            self.credit(i, c, d);
        }
        for x in 0..nl {
            let (k, r) = (usize::from(self.local[x]), self.runs[x]);
            if r == 0 {
                continue;
            }
            // The run's days, from its first session's to its last one's.
            let lo = usize::from(self.day_of[r.trailing_zeros() as usize]);
            let hi = usize::from(self.day_of[63 - r.leading_zeros() as usize]);
            for d in lo..=hi {
                let c = (r & self.on_day[d]).count_ones();
                if c != 0 {
                    self.stats[k * self.ndays + d][3] += i64::from(c);
                }
            }
        }
        mask
    }

    /// Session windows, days and same-day runs, plus padding for `rank`.
    fn prepare(&mut self, ns: usize) -> usize {
        let p = (ns + 1).next_power_of_two().max(8);
        for i in 0..ns {
            self.lowers[i] = self.starts[i] - WEEK;
            self.day_of[i] = day_index(self.starts[i]) as u8;
        }
        self.starts[ns..p].fill(u32::MAX);
        self.lowers[ns..p].fill(u32::MAX);
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
    fn attribute(&mut self, ns: usize, clicks: &[Hit], slow: &SlowIntern) -> u64 {
        let p = self.prepare(ns);
        // Click j is inside the windows of sessions [enter, leave):
        // enter = #(start < ts), leave = #(start - WEEK <= ts).
        self.next.resize(clicks.len(), NONE);
        self.leave.resize(clicks.len(), 0);
        self.first[..=ns].fill(NONE);
        for (j, h) in clicks.iter().enumerate() {
            let a = rank(&self.starts[..p], h.ts);
            let b = rank(&self.lowers[..p], h.ts + 1);
            self.leave[j] = b as u32;
            self.next[j] = self.first[a];
            self.first[a] = j as u32;
        }
        self.epoch += 1;
        let epoch = self.epoch << 32;
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
                let from = if c & !0xffff_ffff == epoch { (c as u32).max(i as u32) } else { i as u32 };
                let to = self.leave[j as usize];
                if to > from {
                    self.covered[k] = epoch | u64::from(to);
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
            let camp = if best_ts >= i64::from(self.lowers[i]) { best_camp } else { self.organic };
            self.credit(i, camp, d);
            mask |= 1 << d;
        }
        mask
    }

    pub(super) fn person(
        &mut self,
        key: u64,
        events: &mut [u64],
        clicks: &mut [Hit],
        cats: &[Vec<u8>],
        slow: &SlowIntern,
        first_day: i32,
    ) {
        if events.is_empty() {
            return;
        }
        // Bucket the events instead of sorting them. Locals keep the slices
        // in registers; a 32-bit time divided by the gap indexes the fixed
        // tables without bounds checks.
        let (mut lo, mut hi) = (usize::MAX, 0);
        let purchases = &mut self.purchases;
        purchases.clear();
        {
            let (bmin, bmax, bcnt) = (&mut *self.bmin, &mut *self.bmax, &mut *self.bcnt);
            let occupied = &mut *self.occupied;
            for &e in events.iter() {
                let ts = timestamp(e);
                let b = (ts / GAP) as usize;
                bmin[b] = bmin[b].min(ts);
                bmax[b] = bmax[b].max(ts);
                bcnt[b] += 1;
                occupied[b / 64] |= 1 << (b % 64);
                lo = lo.min(b);
                hi = hi.max(b);
                let pay = e as u32;
                if pay != 0 {
                    purchases.push((b as u32, pay));
                }
            }
        }
        // Occupied buckets in time order: four positions per word are written
        // unconditionally and the end advances by the word's population, so
        // only words with more than four buckets branch.
        let order = &mut *self.order;
        let mut nb = 0;
        for w in lo / 64..=hi / 64 {
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
        self.cents[..ns].fill(0);
        for &(b, pay) in &self.purchases {
            let i = self.bsess[b as usize] as usize;
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
        let mut mask = 0u64;
        if clicks.is_empty() || ns <= 64 {
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
        } else if ns <= 64 {
            mask = self.attribute_short(ns, clicks, slow);
        } else {
            mask = self.attribute(ns, clicks, slow);
        }
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

/// Nanoseconds a worker spent numbering, scattering and summarizing persons.
/// The stage report in `stream` has no view inside the bins; each worker's
/// split is reported when the report is asked for.
pub(super) struct Phases(pub(super) [u64; 3]);

impl Drop for Phases {
    fn drop(&mut self) {
        if self.0 != [0; 3] && std::env::var_os("STAGES").is_some() {
            let [a, b, c] = self.0.map(|n| n / 1_000_000);
            eprintln!("group phases ms number={a} scatter={b} person={c}");
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

/// Number of entries below x in a sorted slice whose length is a power of
/// two and whose last entry is u32::MAX. The steps are fixed by the length.
#[inline(always)]
fn rank(a: &[u32], x: u32) -> usize {
    let mut base = 0;
    let mut step = a.len() / 2;
    while step > 0 {
        base = if a[base + step - 1] < x { base + step } else { base };
        step /= 2;
    }
    base
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

    #[test]
    fn session_gaps_midnight_and_category_totals() {
        let slow = SlowIntern::new();
        let cats = vec![b"zulu".to_vec(), b"unknown".to_vec()];
        let mut acc = Acc::new(1, 16, cats.len());
        let start = WEEK + DAY - 1;
        let after_midnight = start + GAP;
        let next = after_midnight + GAP + 1;
        let mut events = [
            ev(next + GAP + 1, 1, 0),
            ev(WEEK + 15 * DAY, 0, 0),
            ev(after_midnight, 1, 35),
            ev(next, 0, 110),
            ev(after_midnight, 0, 0),
            ev(start, 0, 75),
            ev(after_midnight, 1, 40),
        ];
        acc.person(41, &mut events, &mut [], &cats, &slow, -2);
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
        let mut clicks = [
            hit(d + 1, b"kfuture", 0),
            dup,
            hit(1, b"katweek", 1),
            hit(a, b"!slow-earliest-id", 3),
            hit(0, b"kexpired", 0),
            hit(a + 1, b"kafter", 4),
            dup,
            hit(a - 1, b"kother", 2),
        ];
        let mut events = [ev(d, 0, 42), ev(b, 0, 20), ev(c, 0, 0), ev(a, 0, 10)];
        let cats = vec![b"book".to_vec()];
        let mut acc = Acc::new(6, 8, cats.len());
        acc.person(1, &mut events, &mut clicks, &cats, &slow, 0);

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
        let mut events = [
            ev(WEEK + 2 * (GAP + 1), 0, 16),
            ev(WEEK, 1, 20),
            ev(WEEK + GAP + 1, 0, 4),
        ];
        acc.person(1, &mut events, &mut [first, later, refresh], &cats, &slow, 7);
        // Camp 0 expires, camp 1 moves to its slot, then gets another click.
        // The two later sessions together tie category 1 on daily revenue.
        assert_eq!(row(&acc.rows[0]), (1, 7, 3, 3, 40, 1));
        acc.person(2, &mut [ev(WEEK, 0, 2)], &mut [first], &cats, &slow, 7);
        acc.person(3, &mut [ev(WEEK, 0, 0)], &mut [], &cats, &slow, 7);
        acc.person(4, &mut [], &mut [later], &cats, &slow, 7);
        assert_eq!(acc.rows.len(), 3);
        assert_eq!(row(&acc.rows[1]), (2, 7, 1, 1, 2, 0));
        assert_eq!(row(&acc.rows[2]), (3, 7, 1, 1, 0, u16::MAX));
        assert_eq!(acc.stats, [[1, 1, 2, 2], [3, 3, 40, 3], [1, 0, 0, 0]]);
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
            let mut clicks: Vec<Hit> = (0..rnd(300))
                .map(|j| Hit { ts: rnd(u64::from(WEEK + span)) as u32, id: 1 + j, camp: rnd(9) as u16 })
                .collect();
            let mut acc_a = Acc::new(10, 18, cats.len());
            acc_a.person(1, &mut events.clone(), &mut clicks.clone(), &cats, &slow, 0);
            // The same history under the long path: 65 isolated sessions are
            // added after the 14 days, which only touch later cells.
            let mut acc_b = Acc::new(10, 18, cats.len());
            for i in 0..65u32 {
                events.push(ev(WEEK + span + i * (GAP + 1) * 2, 0, 0));
            }
            acc_b.person(1, &mut events, &mut clicks, &cats, &slow, 0);
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
