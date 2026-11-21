//! Each event row is written once, straight into its raw user's bin. Dedup
//! sees only a 16-byte candidate per row, partitioned by event ID, and kills
//! losing copies in place. Renaming then compacts each user bin and scatters
//! only renamed rows; neither an all-user interner nor a global sort is needed.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::{histories, Click, Event, Histories};
use crate::clock::{self, ld8, Clock};
use crate::links::Aliases;

/// `n` bytes at `p + at`.
///
/// # Safety
/// The bytes must be readable, and live as long as the result is used.
#[inline(always)]
unsafe fn slice<'b>(p: *const u8, at: u32, n: u32) -> &'b [u8] {
    // SAFETY: forwarded.
    unsafe { std::slice::from_raw_parts(p.add(at as usize), n as usize) }
}

/// The event-row sink: fixed-sequence decode, exact decode for the rest.
struct Events<'a, 'c, 'o> {
    dec: &'a Decoder<'a, 'c>,
    out: &'o mut Scan,
}

impl<'b> wire::Sink<'b, 7> for Events<'_, '_, '_> {
    #[inline(always)]
    fn record(&mut self, f: wire::Fields<'b, 7>) {
        if !self.dec.fast(&f, self.out) {
            self.dec.event_exact(&f.all(), self.out);
        }
    }

    #[inline(always)]
    fn fields(&mut self, f: [&'b [u8]; 7]) {
        self.dec.event_exact(&f, self.out);
    }
}

/// The splitmix64 finalizer. Every step is invertible, so `hash` is a
/// bijection: a mixed key identifies its key exactly, and rows carry mixed
/// keys so later stages index by their bits without hashing again.
#[inline(always)]
pub(super) fn hash(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The inverse of `hash`.
fn unhash(x: u64) -> u64 {
    let x = x ^ (x >> 31) ^ (x >> 62);
    let x = x.wrapping_mul(0x3196_42b2_d24d_8ec3);
    let x = x ^ (x >> 27) ^ (x >> 54);
    let x = x.wrapping_mul(0x96de_1b17_3f11_9089);
    x ^ (x >> 30) ^ (x >> 60)
}

/// One per event row with an ID and a parsable ingest, valid or not: dedup
/// precedes validation. `loc` is `row << user_bits | user_bin` of the row's
/// event in its worker's bins, or DEAD when the row failed validation.
#[derive(Clone, Copy)]
struct Cand {
    h: u64,   // hash(event key): top bits pick the partition
    sec: u32, // ingest seconds; F1 has no fraction
    loc: u32,
}
const DEAD: u32 = u32::MAX;

/// The exact form of a candidate whose ingest or location does not fit.
#[derive(Clone, Copy)]
struct Wide {
    h: u64,
    at: i64,
    at_row: Option<(usize, usize)>,
}

#[derive(Clone, Copy)]
struct Bits {
    user: u32, // raw-user bins = person bins = 1 << user
    id: u32,   // event-id partitions
}

pub(super) fn bins<T>(n: usize) -> Vec<Vec<T>> {
    (0..n).map(|_| Vec::new()).collect()
}

fn sized<T>(n: usize, cap: usize) -> Vec<Vec<T>> {
    (0..n).map(|_| Vec::with_capacity(cap)).collect()
}

/// Push to one of thousands of bins, fetching the line after the tail early.
/// A bin takes a row only every few thousand rows, so its tail's next line
/// is one no hardware prefetcher predicted: a store to it waits in the store
/// buffer for a cold read-for-ownership, and that wait is the scatter's
/// stall. Fetched a line ahead, the line is in cache by the next crossing.
#[inline(always)]
fn push_far<T>(v: &mut Vec<T>, x: T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T1};
        _mm_prefetch(v.as_ptr().add(v.len() + 64 / size_of::<T>()).cast(), _MM_HINT_T1);
    }
    v.push(x);
}

struct Scan {
    ev: Vec<Vec<Event>>,  // by raw user
    cl: Vec<Vec<Click>>,  // by raw user
    cand: Vec<Vec<Cand>>, // by event ID
    wide: Vec<Vec<Wide>>, // by event ID; allocated on first use
    links: Vec<(u64, u64)>,
}

impl Scan {
    #[inline(always)]
    fn candidate(&mut self, id: u64, sec: i64, at_row: Option<(usize, usize)>, b: Bits) {
        let h = hash(id);
        let part = (h >> (64 - b.id)) as usize;
        let loc = match at_row {
            None => DEAD as u64,
            Some((bin, row)) => (row as u64) << b.user | bin as u64,
        };
        if (loc < DEAD as u64 || at_row.is_none()) && (0..=u32::MAX as i64).contains(&sec) {
            push_far(&mut self.cand[part], Cand { h, sec: sec as u32, loc: loc as u32 });
        } else {
            self.push_wide(part, Wide { h, at: sec * 1000, at_row });
        }
    }

    #[cold]
    #[inline(never)]
    fn push_wide(&mut self, part: usize, w: Wide) {
        if self.wide.is_empty() {
            self.wide = bins(self.cand.len());
        }
        self.wide[part].push(w);
    }
}

const VIEW: u32 = u32::from_le_bytes(*b"view");
const CART: u32 = u32::from_le_bytes(*b"cart");
const PURCHASE: u64 = u64::from_le_bytes(*b"purchase");
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;
const ZEROS: u64 = 0x3030_3030_3030_3030;

/// Event-ID key, for equality and hashing only (never order). `e` and seven
/// or eight more bytes in 1..=127, the generator's shape, pack without a
/// gather: bit 63, then the suffix bytes. Their last byte is not NUL, so bits
/// 48..=62 are never all clear, unlike any interned key (bits 32..=62 clear);
/// packed keys clear bit 63. Seven and eight bytes differ in byte 7.
#[inline(always)]
fn event_key(s: &Shared, b: &[u8]) -> Option<u64> {
    let n = b.len();
    if let [b'e', .., _] = b {
        if let Some(tail) = b.last_chunk::<8>().filter(|_| n == 8 || n == 9) {
            let w = u64::from_le_bytes(*tail) >> (8 * (9 - n));
            let pad = u64::from(n == 8) << 56;
            if ((w | pad) | (w | pad).wrapping_sub(ONES)) & HIGH == 0 {
                return Some(w | 1 << 63);
            }
        }
    }
    id_key(s, b)
}

/// Eight bytes of `buf` from the start of `field`, a subslice of `buf`, when
/// that many remain. Reading past the field's end stays inside the chunk.
#[inline(always)]
fn word_at(buf: &[u8], field: &[u8]) -> Option<u64> {
    let at = field.as_ptr() as usize - buf.as_ptr() as usize;
    Some(u64::from_le_bytes(*buf.get(at..)?.first_chunk::<8>()?))
}

/// `canonical_number` of the first `n` (1..=8) bytes of `w`, without a loop.
#[inline(always)]
fn canonical8(w: u64, n: usize) -> Option<u32> {
    debug_assert!((1..=8).contains(&n));
    // Move the digits to the top lanes and pad the bottom with '0'.
    let shift = 8 * (8 - n) as u32;
    let x = (w << shift) | (ZEROS & !(u64::MAX << shift));
    let d = x.wrapping_sub(ZEROS);
    // The lowest out-of-range lane sets a high bit; valid lanes never carry.
    if (d | x.wrapping_add(0x4646_4646_4646_4646)) & HIGH != 0 || (n > 1 && w as u8 == b'0') {
        return None;
    }
    Some(crate::clock::decimal8(d) as u32)
}

/// Rows per decode batch. The scanner invokes its callback from several
/// sites; a callback that only stores fields stays small enough to inline,
/// and each decoder runs as one loop with its constants in registers.
const BATCH: usize = 32;

#[inline(always)]
fn batched<'b, const N: usize>(buf: &'b [u8], mut decode: impl FnMut(&[[&'b [u8]; N]])) {
    let mut batch = [[&buf[..0]; N]; BATCH];
    let mut n = 0;
    wire::rows::<N>(buf, |f| {
        batch[n % BATCH] = f;
        n += 1;
        if n == BATCH {
            decode(&batch);
            n = 0;
        }
    });
    decode(&batch[..n]);
}

/// Everything a row decoder reads, hoisted out of the shared context.
struct Decoder<'a, 'c> {
    cx: &'a Ctx<'c>,
    clock: &'a Clock,
    s: &'a Shared,
    origin: i64,
    start: i64, // event window; clicks reach back a week
    span: u64,
    click_start: i64,
    click_span: u64,
    b: Bits,
}

impl Decoder<'_, '_> {
    fn new<'a, 'c>(cx: &'a Ctx<'c>, clock: &'a Clock, s: &'a Shared, origin: i64, b: Bits) -> Decoder<'a, 'c> {
        let w = cx.window;
        Decoder { cx, clock, s, origin, start: w.start, span: (w.end - w.start) as u64,
                  click_start: w.click_start, click_span: (w.end - w.click_start) as u64, b }
    }

    /// The decode of a clean event row of the usual shapes: ids of eight or
    /// nine bytes, an F1 ingest with a cached date, a ts of the three forms'
    /// usual lengths. Fields are sliced by arithmetic from the separator
    /// offsets, one at a time so that few values stay live, and F1 and F3
    /// (the unpredictable pair) are both computed and the row's selected by
    /// mask. Returns false for any other row, which then takes the exact
    /// path; both compute the same result.
    #[inline(always)]
    fn fast(&self, f: &wire::Fields<'_, 7>, out: &mut Scan) -> bool {
        let p = f.first();
        // SAFETY: `Fields` guarantees p[0..=f.end(6)] readable. A quote flag
        // reads a field's first byte, at most the LF. Every other load is
        // bounded by a later field once that field's length is known: see
        // each.
        unsafe {
            // In a clean row a field starts with a quote exactly when it also
            // ends with one; the content lies between.
            let quoted = |at: u32| u32::from(*p.add(at as usize) == b'"');
            // Event id as in `event_key`: the last eight bytes, the prefix
            // shifted out of an eight-byte id.
            let e0 = f.end(0);
            let q0 = quoted(0);
            let l0 = e0 - 2 * q0;
            if l0.wrapping_sub(8) > 1 {
                return false;
            }
            let w = ld8(p.add((e0 - q0 - 8) as usize)) >> (8 * (9 - l0));
            let pad = u64::from(l0 == 8) << 56;
            if !((*p.add(q0 as usize) == b'e') & (((w | pad) | (w | pad).wrapping_sub(ONES)) & HIGH == 0)) {
                return false;
            }
            // User id.
            let r1 = e0 + 1;
            let e1 = f.end(1);
            let q1 = quoted(r1);
            let l1 = e1 - r1 - 2 * q1;
            if l1.wrapping_sub(8) > 1 {
                return false;
            }
            let (ukey, u_ok) = atom::pack_raw(p.add((r1 + q1) as usize), l1);
            if !u_ok {
                return false;
            }
            // Ingest first: its 20 bytes bound the ts loads that follow.
            let r2 = e1 + 1;
            let e2 = f.end(2);
            let q2 = quoted(r2);
            let l2 = e2 - r2 - 2 * q2;
            let r3 = e2 + 1;
            let e3 = f.end(3);
            let q3 = quoted(r3);
            if e3 - r3 - 2 * q3 != 20 {
                return false;
            }
            let g = p.add((r3 + q3) as usize);
            let (sec, in_ok) = self.clock.iso_raw(g);
            if !(in_ok & (*g.add(19) == b'Z')) {
                return false;
            }
            // The ts: F2 is rare enough for a branch; F1 and F3 are computed
            // both and selected. Their loads reach at most 25 bytes past the
            // content start, inside the ingest.
            let t = p.add((r2 + q2) as usize);
            let (ts, ts_ok) = match self.clock.parse(slice(p, r2 + q2, l2)) { Some(v) => (v, true), None => (0, false) };
            // Type: the eight bytes ending at the field's end hold a 4- or
            // 8-byte word; they start inside the row, after the ingest.
            let r4 = e3 + 1;
            let e4 = f.end(4);
            let q4 = quoted(r4);
            let l4 = e4 - r4 - 2 * q4;
            let tw = ld8(p.add((e4 - q4 - 8) as usize)).wrapping_shr(8u32.wrapping_sub(l4).wrapping_mul(8));
            let purchase = (l4 == 8) & (tw == PURCHASE);
            let type_ok = purchase | ((l4 == 4) & ((tw as u32 == VIEW) | (tw as u32 == CART)));
            let r5 = e4 + 1;
            let e5 = f.end(5);
            let q5 = quoted(r5);
            let l5 = e5 - r5 - 2 * q5;
            let at = sec * 1000;
            let mut valid = ts_ok & type_ok & (l5 != 0)
                & ((ts.wrapping_sub(self.start) as u64) < self.span) & (at.wrapping_sub(ts) <= DAY_MS);
            let mut pay = 0u32;
            if purchase & valid {
                let r6 = e5 + 1;
                let q6 = quoted(r6);
                let l6 = f.end(6) - r6 - 2 * q6;
                match parse_price(slice(p, r6 + q6, l6)) {
                    Some(c) => pay = c | u32::from(product_cat(self.cx, slice(p, r5 + q5, l5))) << 24,
                    None => valid = false,
                }
            }
            let key = hash(ukey);
            let bin = (key >> (64 - self.b.user)) as usize;
            let val = (ts.wrapping_sub(self.origin) as u64) << 32 | u64::from(pay);
            let mut at_row = None;
            if valid {
                let v = &mut out.ev[bin];
                at_row = Some((bin, v.len()));
                push_far(v, Event { key, val });
            }
            out.candidate(w | 1 << 63, sec, at_row, self.b);
        }
        true
    }

    #[inline(never)]
    fn event_exact(&self, f: &[&[u8]; 7], out: &mut Scan) {
        self.event(f, out);
    }

    #[inline(always)]
    fn event(&self, f: &[&[u8]; 7], out: &mut Scan) {
        let Some(id) = event_key(self.s, f[0]) else { return };
        let Some(sec) = self.clock.ingest_s(f[3]) else { return };
        let mut at_row = None;
        if let Some(val) = self.payload(f, sec * 1000) {
            // Mixed until renaming.
            let key = hash(user_key(self.s, f[1]));
            let bin = (key >> (64 - self.b.user)) as usize;
            let v = &mut out.ev[bin];
            at_row = Some((bin, v.len()));
            push_far(v, Event { key, val });
        }
        out.candidate(id, sec, at_row, self.b);
    }

    #[inline(always)]
    fn payload(&self, f: &[&[u8]; 7], at: i64) -> Option<u64> {
        // Whole-word compares: a view/cart split costs no branch.
        let purchase = match *f[4] {
            [a, b, c, d] => {
                let w = u32::from_le_bytes([a, b, c, d]);
                if w != VIEW && w != CART { return None; }
                false
            }
            [a, b, c, d, e, g, h, i] if u64::from_le_bytes([a, b, c, d, e, g, h, i]) == PURCHASE => true,
            _ => return None,
        };
        if f[1].is_empty() || f[5].is_empty() { return None; }
        let ts = self.clock.parse(f[2])?;
        // One unsigned compare per bound pair; no instant overflows i64 here.
        if ts.wrapping_sub(self.start) as u64 >= self.span || at - ts > DAY_MS {
            return None;
        }
        let pay = if purchase { parse_price(f[6])? | u32::from(product_cat(self.cx, f[5])) << 24 } else { 0 };
        Some(((ts - self.origin) as u64) << 32 | u64::from(pay))
    }

    #[inline(never)]
    fn clicks(&self, buf: &[u8], rows: &[[&[u8]; 4]], out: &mut Scan) {
        for f in rows {
            self.click(buf, f, out);
        }
    }

    // All copies of a click are identical. Window membership is existential
    // per campaign and credit is a min/max over keys. Both are idempotent:
    // duplicating an identical tuple cannot change either, so clicks skip dedup.
    #[inline(always)]
    fn click(&self, buf: &[u8], f: &[&[u8]; 4], out: &mut Scan) {
        if f[0].is_empty() || f[1].is_empty() { return; }
        let Some(ts) = self.clock.parse(f[3]) else { return };
        if ts.wrapping_sub(self.click_start) as u64 >= self.click_span { return; }
        let Some(camp) = self.campaign(buf, f[2]) else { return };
        let key = hash(user_key(self.s, f[1]));
        let hit = Hit { ts: (ts - self.origin) as u32, id: user_key(self.s, f[0]), camp };
        push_far(&mut out.cl[(key >> (64 - self.b.user)) as usize], Click { key, hit });
    }

    /// `campaign_of` with the number decoded in one word.
    #[inline(always)]
    fn campaign(&self, buf: &[u8], f: &[u8]) -> Option<u16> {
        if let [b'c', b'_', digits @ ..] = f {
            if (1..=8).contains(&digits.len()) {
                if let Some(n) = word_at(buf, digits).and_then(|w| canonical8(w, digits.len())) {
                    if let Some(&v) = self.cx.camp_by_num.get(n as usize) {
                        return (v >= 0).then_some(v as u16);
                    }
                    return self.cx.camp_other.get(f).copied();
                }
            }
        }
        campaign_of(self.cx, f)
    }
}

/// Every worker's event bins, writable from dedup jobs.
struct Rows {
    ptr: Vec<*mut Event>, // worker * bins + bin
    len: Vec<usize>,
    bits: u32,
}

// SAFETY: a row is written only through `kill`, by the one dedup job that owns
// its event ID's partition (each row has exactly one candidate). The vectors
// are neither read, moved nor resized while dedup jobs run.
unsafe impl Sync for Rows {}

impl Rows {
    fn new(evs: &mut [Vec<Vec<Event>>], bits: u32) -> Self {
        let ptr = evs.iter_mut().flatten().map(|v| v.as_mut_ptr()).collect();
        let len = evs.iter().flatten().map(Vec::len).collect();
        Self { ptr, len, bits }
    }

    #[inline(always)]
    fn kill(&self, worker: usize, loc: u32) {
        if loc != DEAD {
            let bin = loc as usize & ((1 << self.bits) - 1);
            self.kill_at(worker, bin, (loc >> self.bits) as usize);
        }
    }

    #[inline(always)]
    fn kill_at(&self, worker: usize, bin: usize, row: usize) {
        let i = (worker << self.bits) | bin;
        assert!(row < self.len[i]);
        // SAFETY: in bounds by the assert; exclusive per the Sync argument.
        unsafe { (*self.ptr[i].add(row)).key = 0 };
    }
}

// Copies of one id may sit in different workers' parts. Part p holds worker
// p's candidates, so a loser's location is resolved against worker p's bins.
fn dedup(parts: Vec<Vec<Cand>>, wide: Vec<Vec<Wide>>, rows: &Rows, id_bits: u32) {
    if wide.iter().any(|w| !w.is_empty()) {
        return dedup_exact(parts, wide, rows);
    }
    let n: usize = parts.iter().map(Vec::len).sum();
    if n == 0 { return; }
    // Two bits per hash prefix. A prefix seen once cannot be a duplicate, so
    // those rows skip the id table. A shared prefix is only a suspect: the
    // exact table still compares full ids. The partition fixed the top bits;
    // the bits below them pick the word and the bit.
    let words = n.div_ceil(4).next_power_of_two();
    let wbits = words.trailing_zeros();
    let probe = |h: u64| {
        let g = h << id_bits;
        ((g >> (63 - wbits) >> 1) as usize, 1u64 << ((g >> (58 - wbits)) & 63))
    };
    let mut marks = vec![[0u64; 2]; words]; // present, repeated
    for c in parts.iter().flatten() {
        let (i, bit) = probe(c.h);
        let m = &mut marks[i];
        m[1] |= m[0] & bit;
        m[0] |= bit;
    }
    let mut suspects = n;
    for m in &marks {
        suspects -= (m[0] ^ m[1]).count_ones() as usize;
    }
    if suspects == 0 { return; }
    // Slot value: part index + 1 in the high half, row in the low half. The
    // low hash bits are independent of the prefix bits used above.
    let mask = (suspects * 2).next_power_of_two() - 1;
    let mut table = vec![0u64; mask + 1];
    for (p, part) in parts.iter().enumerate() {
        for (i, c) in part.iter().enumerate() {
            let (w, bit) = probe(c.h);
            if marks[w][1] & bit == 0 { continue; }
            let me = ((p as u64 + 1) << 32) | i as u64;
            let mut s = c.h as usize & mask;
            loop {
                let ptr = table[s];
                if ptr == 0 {
                    table[s] = me;
                    break;
                }
                let (q, j) = ((ptr >> 32) as usize - 1, ptr as u32 as usize);
                let old = parts[q][j];
                if old.h == c.h {
                    // Copies with equal ingest are identical: keep either.
                    if c.sec < old.sec {
                        rows.kill(q, old.loc);
                        table[s] = me;
                    } else {
                        rows.kill(p, c.loc);
                    }
                    break;
                }
                s = (s + 1) & mask;
            }
        }
    }
}

/// The same contest at full width, for partitions holding any wide candidate.
#[cold]
#[inline(never)]
fn dedup_exact(parts: Vec<Vec<Cand>>, wide: Vec<Vec<Wide>>, rows: &Rows) {
    let mut best: HashMap<u64, (i64, usize, Option<(usize, usize)>)> = HashMap::new();
    let compact = parts.iter().enumerate().flat_map(|(w, p)| p.iter().map(move |c| {
        let at_row = (c.loc != DEAD).then(|| (c.loc as usize & ((1 << rows.bits) - 1), (c.loc >> rows.bits) as usize));
        (w, c.h, i64::from(c.sec) * 1000, at_row)
    }));
    let full = wide.iter().enumerate().flat_map(|(w, v)| v.iter().map(move |x| (w, x.h, x.at, x.at_row)));
    for (w, h, at, at_row) in compact.chain(full) {
        let kill = |w: usize, at_row: Option<(usize, usize)>| {
            if let Some((bin, row)) = at_row { rows.kill_at(w, bin, row); }
        };
        match best.entry(h) {
            std::collections::hash_map::Entry::Vacant(e) => { e.insert((at, w, at_row)); }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let old = *e.get();
                if at < old.0 {
                    kill(old.1, old.2);
                    e.insert((at, w, at_row));
                } else {
                    kill(w, at_row);
                }
            }
        }
    }
}

/// Mixed raw user -> (person name, its bin) for one user bin. Each raw user
/// recurs about a dozen times per bin, and a hit here is predictable where the
/// alias probe's hit-or-miss outcome is not. Keys in one bin share their top
/// bits, so the low bits index.
#[derive(Default)]
struct Names {
    table: Vec<(u64, u64, usize)>,
    used: usize,
}

impl Names {
    fn reset(&mut self, rows: usize) {
        self.table.clear();
        // A raw user recurs about a dozen times per bin: start near 25% load.
        self.table.resize((rows / 4).max(64).next_power_of_two(), (0, 0, 0));
        self.used = 0;
    }

    #[inline(always)]
    fn get(&mut self, key: u64, aliases: &Aliases, shift: u32) -> (u64, usize) {
        let mask = self.table.len() - 1;
        let mut i = key as usize & mask;
        loop {
            let (k, name, bin) = self.table[i];
            if k == key { return (name, bin); }
            if k == 0 { return self.insert(i, key, aliases, shift); }
            i = (i + 1) & mask;
        }
    }

    #[cold]
    #[inline(never)]
    fn insert(&mut self, i: usize, key: u64, aliases: &Aliases, shift: u32) -> (u64, usize) {
        let name = aliases.resolve(unhash(key));
        let bin = (hash(name) >> shift) as usize;
        self.table[i] = (key, name, bin);
        self.used += 1;
        if self.used * 2 > self.table.len() {
            let grown = vec![(0, 0, 0); self.table.len() * 4];
            let old = std::mem::replace(&mut self.table, grown);
            let mask = self.table.len() - 1;
            for e in old.into_iter().filter(|e| e.0 != 0) {
                let mut j = e.0 as usize & mask;
                while self.table[j].0 != 0 { j = (j + 1) & mask; }
                self.table[j] = e;
            }
        }
        (name, bin)
    }
}

/// Per-worker scratch for renaming.
#[derive(Default)]
struct Renamer {
    names: Names,
    ev: Vec<(usize, Event)>,
    cl: Vec<(usize, Click)>,
}

/// Name every row of one part: rows whose person lives in this bin are
/// compacted in place, the rest are staged with their bin. Both are written
/// unconditionally and only the cursors move, so the split costs no branch.
#[inline(always)]
fn split<T: Copy>(rows: &mut Vec<T>, stage: &mut Vec<(usize, T)>, bin: usize,
                  mut name: impl FnMut(&T) -> Option<(usize, T)>) {
    let n = rows.len();
    stage.clear();
    stage.reserve(n);
    let (mut kept, mut moved) = (0, 0);
    let (rp, sp) = (rows.as_mut_ptr(), stage.as_mut_ptr());
    for i in 0..n {
        // SAFETY: i < n.
        let Some((to, r)) = name(unsafe { &*rp.add(i) }) else { continue };
        // SAFETY: kept <= i < n, and moved <= i < n <= stage capacity.
        unsafe {
            *rp.add(kept) = r;
            *sp.add(moved) = (to, r);
        }
        kept += usize::from(to == bin);
        moved += usize::from(to != bin);
    }
    rows.truncate(kept);
    // SAFETY: the first `moved` staged rows were written above.
    unsafe { stage.set_len(moved) };
}

// Every raw user in this bin shares its top hash bits with exactly one alias
// shard, so a resolve stays in a cache-sized table; each happens once per raw
// user. A user named by itself already lives in its person's bin: such rows
// stay in place and only renamed rows are scattered. Dedup losers (key 0) drop.
fn rename(mut parts: Vec<Histories>, aliases: &Aliases, dest: &mut [Histories], r: &mut Renamer, shift: u32, bin: usize) -> Vec<Histories> {
    let rows = parts.iter().map(|p| p.ev.len() + p.cl.len()).sum();
    let names = &mut r.names;
    names.reset(rows);
    for p in &mut parts {
        split(&mut p.ev, &mut r.ev, bin, |e| {
            if e.key == 0 { return None; }
            let (key, to) = names.get(e.key, aliases, shift);
            Some((to, Event { key, val: e.val }))
        });
        for &(to, e) in &r.ev {
            dest[to].ev.push(e);
        }
        split(&mut p.cl, &mut r.cl, bin, |c| {
            let (key, to) = names.get(c.key, aliases, shift);
            Some((to, Click { key, hit: c.hit }))
        });
        for &(to, c) in &r.cl {
            dest[to].cl.push(c);
        }
    }
    debug_assert!(parts.iter().all(|p| p.ev.iter().all(|e| (hash(e.key) >> shift) as usize == bin)));
    parts
}

// One owner for each job. The lock is acquired once per shard, never per row.
pub(super) fn consume<J:Send,S:Send>(jobs:Vec<J>,workers:usize, init:impl Fn()->S+Sync,
                            run:impl Fn(&mut S,J)+Sync)->Vec<S> {
    let queue=Mutex::new(jobs.into_iter());
    thread::scope(|scope| {
        let handles: Vec<_>=(0..workers).map(|_|scope.spawn(||{
            let mut s=init();
            loop {
                let job=queue.lock().unwrap().next();
                match job { Some(j)=>run(&mut s,j),None=>break }
            }
            s
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).collect()
    })
}

pub(super) fn compute(data:&Path, threads:usize)->Result<Tables,Box<dyn std::error::Error>> {
    let total=Instant::now();
    let w=load_window(&data.join("manifest.json"))?;
    let origin=w.start_day as i64*DAY_MS-WEEK_MS;
    if w.end<=w.start || w.end-origin>u32::MAX as i64 {return Err("window does not fit compact millisecond offsets".into());}
    let days=((w.end-1).div_euclid(DAY_MS)-w.start_day as i64+1) as usize;
    let clock=Clock::new(w.start,w.end);
    let (prod_cat,unknown,prod_other,cat_names)=load_products(&data.join("products.csv"))?;
    let (camp_by_num,mut camps,camp_other)=load_campaigns(&data.join("campaigns.csv"))?;
    let organic=u16::try_from(camps.len())?;
    camps.push(Camp{name:b"organic".to_vec(),channel:b"organic".to_vec()});
    let cx=Ctx{prod_cat:&prod_cat,unknown,prod_other:&prod_other,camp_by_num:&camp_by_num,camp_other:&camp_other,window:&w};
    let mut maps=Vec::<Mmap>::new();
    let mut chunks=Vec::new();
    let mut bytes=[0usize;3];
    for (kind,dir) in [(0u8,"events"),(1,"clicks"),(2,"links")] {
        for p in list_csv(&data.join(dir))? {
            let f=File::open(p)?;
            if f.metadata()?.len()==0 {continue;}
            let m=unsafe{Mmap::map(&f)?};
            let _=m.advise(Advice::Sequential);
            bytes[kind as usize]+=m.len();
            chunks.extend(split_chunks(maps.len(),kind,&m));
            maps.push(m);
        }
    }
    let event_bytes=bytes[0];
    let threads=threads.max(1).min(chunks.len().max(1));
    let eb=(event_bytes / (77 * 16384)).next_power_of_two().clamp(8,2048);
    let pb=(event_bytes / (77 * 65536)).next_power_of_two().clamp(16,512);
    let b=Bits{user:pb.trailing_zeros(),id:eb.trailing_zeros()};
    chunks.sort_unstable_by_key(|c|std::cmp::Reverse(c.end-c.start));
    let next=AtomicUsize::new(0);
    let shared=Shared{slow:Mutex::new(SlowIntern::new())};
    let dec=Decoder::new(&cx,&clock,&shared,origin,b);
    let prep=total.elapsed();
    // Workers take chunks dynamically; leave headroom over an even share.
    let share=|bytes:usize,row:usize,n:usize|bytes/row/threads*5/4/n+64;
    let scans=thread::scope(|scope|{
        let handles:Vec<_>=(0..threads).map(|_|scope.spawn(||{
            let mut out=Scan{ev:sized(pb,share(event_bytes,77,pb)),cl:sized(pb,share(bytes[1],47,pb)),
                cand:sized(eb,share(event_bytes,77,eb)),wide:Vec::new(),links:Vec::new()};
            loop {
                let i=next.fetch_add(1,AtomicOrdering::Relaxed);
                let Some(c)=chunks.get(i) else{break};
                let buf=&maps[c.file as usize][c.start..c.end];
                match c.kind {
                    0=>{wire::records::<7, _>(buf,Events{dec:&dec,out:&mut out});}
                    1=>batched::<4>(buf,|rows|dec.clicks(buf,rows,&mut out)),
                    _=>wire::rows::<2>(buf,|[a,b]| {
                        if !a.is_empty() && !b.is_empty() && a!=b {
                            out.links.push((user_key(&shared,a),user_key(&shared,b)));
                        }
                    }),
                }
            }
            (out,total.elapsed())
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).collect::<Vec<_>>()
    });
    let parsed=total.elapsed();
    // Idle tail of the parse: the first worker to run out of chunks waits this long.
    let tail=parsed.saturating_sub(scans.iter().map(|s|s.1).min().unwrap_or(parsed));
    let scans=scans.into_iter().map(|s|s.0);
    let slow=shared.slow.into_inner().unwrap();
    let mut evs=Vec::with_capacity(threads);
    let mut cls=Vec::with_capacity(threads);
    let mut cands:Vec<(Vec<Vec<Cand>>,Vec<Vec<Wide>>)>=(0..eb).map(|_|(Vec::new(),Vec::new())).collect();
    let mut lks=Vec::with_capacity(threads);
    for mut s in scans {
        evs.push(s.ev);
        cls.push(s.cl);
        s.wide.resize_with(eb,Vec::new);
        for ((c,wd),(cv,wv)) in cands.iter_mut().zip(s.cand.into_iter().zip(s.wide)) {c.push(cv);wd.push(wv);}
        lks.push(s.links);
    }
    // Linking and dedup are independent. Side by side, dedup jobs fill the
    // cores that the link build's serial steps leave idle.
    let rows=Rows::new(&mut evs,b.user);
    let (aliases,indexed)=thread::scope(|scope|{
        let linker=scope.spawn(||{let a=links::build(lks,&slow,threads,pb);(a,total.elapsed())});
        consume(cands,threads,||(),|_,(c,wd)|dedup(c,wd,&rows,b.id));
        linker.join().unwrap()
    });
    drop(rows);
    let reduced=total.elapsed();
    let shift=64-b.user;
    let hist=||(0..pb).map(|_| Histories::default()).collect::<Vec<_>>();
    let mut by_user:Vec<Vec<Histories>>=bins(pb);
    for (ev,cl) in evs.into_iter().zip(cls) {
        for (dst,(ev,cl)) in by_user.iter_mut().zip(ev.into_iter().zip(cl)) {dst.push(Histories{ev,cl});}
    }
    let named=consume(by_user.into_iter().enumerate().collect(),threads,||(hist(),Vec::new(),Renamer::default()),
        |(out,kept,r),(b,job)|kept.push((b,rename(job,&aliases,out,r,shift,b))));
    let renamed=total.elapsed();
    let mut by_name:Vec<Vec<Histories>>=bins(pb);
    for (r,kept,_) in named {
        for (dst,src) in by_name.iter_mut().zip(r) {dst.push(src);}
        for (b,mut parts) in kept {by_name[b].append(&mut parts);}
    }
    by_name.sort_unstable_by_key(|s|std::cmp::Reverse(s.iter().map(|p|p.ev.len()+p.cl.len()).sum::<usize>()));
    let partials=consume(by_name,threads,||Acc::new(camps.len(),days,cat_names.len()),
                         |acc,job|histories(job,acc,&cat_names,&slow,w.start_day));
    let grouped=total.elapsed();
    let mut users=Vec::with_capacity(partials.iter().map(|p|p.rows.len()).sum());
    let mut stats=vec![[0i64;4];camps.len()*days];
    for mut p in partials {
        users.append(&mut p.rows);
        for (t,v) in stats.iter_mut().zip(p.stats) {
            for i in 0..4{t[i]+=v[i];}
        }
    }
    let mut camps_out=Vec::new();
    for (i,[sessions,purchasing,revenue,assisted]) in stats.into_iter().enumerate(){
        if sessions!=0 || assisted!=0{
            camps_out.push(CampRow{camp:(i/days) as u16,day:w.start_day+(i%days) as i32,sessions,purchasing,revenue,assisted});
        }
    }
    if std::env::var_os("STAGES").is_some() {
        eprintln!("ms prep={} parse={} links={} dedup={} rename={} group={} merge={} parse_tail={}{}",
            prep.as_millis(),(parsed-prep).as_millis(),(indexed-parsed).as_millis(),(reduced-indexed).as_millis(),
            (renamed-reduced).as_millis(),(grouped-renamed).as_millis(),(total.elapsed()-grouped).as_millis(),
            tail.as_millis(),cpu_stats());
    }
    // The process exits right after writing; unmapping is wasted time.
    std::mem::forget(maps);
    std::mem::forget(aliases);
    Ok(Tables{users,camps_out,camps,cat_names,slow,organic})
}

/// CPU seconds and minor faults of this process so far (Linux), for STAGES.
fn cpu_stats()->String {
    let Ok(stat)=fs::read_to_string("/proc/self/stat") else {return String::new()};
    let Some((_,rest))=stat.rsplit_once(')') else {return String::new()};
    let f:Vec<&str>=rest.split_whitespace().collect();
    let num=|i:usize|f.get(i).and_then(|s|s.parse::<u64>().ok()).unwrap_or(0);
    // Fields after the command: state is index 0, so minflt=7, utime=11, stime=12 (ticks of 10 ms).
    format!(" cpu_user_ms={} cpu_sys_ms={} minflt={}",num(11)*10,num(12)*10,num(7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_a_bijection() {
        let mut x = 0x0123_4567_89ab_cdefu64;
        for i in 0..100_000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            for k in [x, i, u64::MAX - i, 1 << (i % 64)] {
                assert_eq!(unhash(hash(k)), k);
            }
        }
        assert_eq!(hash(0), 0);
    }

    #[test]
    fn event_keys_are_exact() {
        let s = Shared { slow: Mutex::new(SlowIntern::new()) };
        let mut seen = HashMap::new();
        let mut ids: Vec<Vec<u8>> = Vec::new();
        for n in 0..=10usize {
            for fill in [b'0', b'z', 1, 0x7f, 0x80, 0] {
                for first in [b'e', b'E', b'u', 0] {
                    let mut v = vec![fill; n];
                    if n > 0 { v[0] = first; }
                    if n > 2 { v[n - 1] = b'9'; v[1] = b'a'; }
                    ids.push(v);
                }
            }
        }
        ids.push(b"e\0\0\0\0\0\0\0@".to_vec());
        ids.push(b"e@\0\0\0\0\0\0".to_vec());
        for id in &ids {
            let Some(k) = event_key(&s, id) else { assert!(id.is_empty()); continue };
            if let Some(prev) = seen.insert(k, id.clone()) {
                assert_eq!(&prev, id, "key collision");
            }
        }
    }

    #[test]
    fn canonical8_matches_canonical_number() {
        let mut cases: Vec<Vec<u8>> = Vec::new();
        for n in 1..=8usize {
            for d in [b'0', b'1', b'9', b'/', b':', 0x80, 0xff, b' '] {
                for at in 0..n {
                    let mut v = b"12345678"[..n].to_vec();
                    v[at] = d;
                    cases.push(v);
                }
            }
        }
        for v in cases {
            let mut w = [b','; 8];
            w[..v.len()].copy_from_slice(&v);
            assert_eq!(canonical8(u64::from_le_bytes(w), v.len()), canonical_number(&v), "{v:?}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    // cargo test --release --bin solve bench_parse -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_parse() {
        let data = Path::new("/tmp/ss-20m");
        let w = load_window(&data.join("manifest.json")).unwrap();
        let origin = w.start_day as i64 * DAY_MS - WEEK_MS;
        let clock = Clock::new(w.start, w.end);
        let (prod_cat, unknown, prod_other, _names) = load_products(&data.join("products.csv")).unwrap();
        let (camp_by_num, _camps, camp_other) = load_campaigns(&data.join("campaigns.csv")).unwrap();
        let cx = Ctx { prod_cat: &prod_cat, unknown, prod_other: &prod_other, camp_by_num: &camp_by_num,
                       camp_other: &camp_other, window: &w };
        let buf = fs::read(data.join("events/part-00043.csv")).unwrap();
        let body = &buf[memchr::memchr(b'\n', &buf).unwrap() + 1..];
        let rows_n = memchr::memchr_iter(b'\n', body).count();
        let shared = Shared { slow: Mutex::new(SlowIntern::new()) };
        let b = Bits { user: 9, id: 11 };
        let dec = Decoder::new(&cx, &clock, &shared, origin, b);
        let only = std::env::var("BENCH_ONLY").ok();
        let report = |name: &str, f: &mut dyn FnMut()| {
            if only.as_deref().is_some_and(|o| o != name) { return; }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!("{name:>12}: {:6.2} ns/row {:6.2} GB/s", best * 1e9 / rows_n as f64, body.len() as f64 / best / 1e9);
        };
        report("masks", &mut || {
            let mut acc = 0u64;
            let mut at = 0;
            while body.len() - at >= 64 {
                let (c, n, q) = unsafe { crate::wire::masks64_for_bench(body.as_ptr().add(at)) };
                acc = acc.wrapping_add(c ^ n.rotate_left(7) ^ q.rotate_left(13));
                at += 64;
            }
            std::hint::black_box(acc);
        });
        report("rows<7>", &mut || {
            let mut acc = 0usize;
            wire::rows::<7>(body, |f| acc += f[0].len() + f[6].len());
            std::hint::black_box(acc);
        });
        report("batch", &mut || {
            let mut acc = 0usize;
            let mut batch = [[&body[..0]; 7]; BATCH];
            let mut n = 0;
            #[inline(never)]
            fn sink(rows: &[[&[u8]; 7]], acc: &mut usize) { for f in rows { *acc += f[0].len() + f[6].len(); } }
            wire::rows::<7>(body, |f| {
                batch[n % BATCH] = f;
                n += 1;
                if n == BATCH { sink(&batch, &mut acc); n = 0; }
            });
            sink(&batch[..n], &mut acc);
            std::hint::black_box(acc);
        });
        report("+id", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| { if let Some(id) = id_key(&shared, f[0]) { acc ^= id; } });
            std::hint::black_box(acc);
        });
        report("+ingest", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| {
                let Some(id) = id_key(&shared, f[0]) else { return };
                let Some(at) = clock.ingest(f[3]) else { return };
                acc ^= id ^ at as u64;
            });
            std::hint::black_box(acc);
        });
        report("+payload", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| {
                let Some(id) = id_key(&shared, f[0]) else { return };
                let Some(at) = clock.ingest(f[3]) else { return };
                let v = dec.payload(&f, at).unwrap_or(0);
                acc ^= id ^ at as u64 ^ v;
            });
            std::hint::black_box(acc);
        });
        report("+user", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| {
                let Some(id) = id_key(&shared, f[0]) else { return };
                let Some(at) = clock.ingest(f[3]) else { return };
                let v = dec.payload(&f, at).unwrap_or(0);
                let u = if f[1].is_empty() { 0 } else { user_key(&shared, f[1]) };
                acc ^= id ^ at as u64 ^ v ^ u;
            });
            std::hint::black_box(acc);
        });
        report("offsets", &mut || {
            struct S(u32);
            impl<'b> wire::Sink<'b, 7> for S {
                #[inline(always)]
                fn record(&mut self, f: wire::Fields<'b, 7>) {
                    let (p, o) = (f.first(), f.ends());
                    for i in 0..7 {
                        self.0 += o[i] + unsafe { u32::from(*p.add(if i == 0 { 0 } else { o[i - 1] as usize + 1 }) == b'"') };
                    }
                }
                fn fields(&mut self, f: [&'b [u8]; 7]) { self.0 += f[6].len() as u32; }
            }
            let s = wire::records::<7, _>(body, S(0));
            std::hint::black_box(s.0);
        });
        let (mut slow, mut exact) = (0usize, 0usize);
        report("scan_events", &mut || {
            struct S<'a, 'c, 'o> { dec: &'a Decoder<'a, 'c>, out: &'o mut Scan, slow: usize, exact: usize }
            impl<'b> wire::Sink<'b, 7> for S<'_, '_, '_> {
                #[inline(always)]
                fn record(&mut self, f: wire::Fields<'b, 7>) {
                    if !self.dec.fast(&f, self.out) { self.slow += 1; self.dec.event_exact(&f.all(), self.out); }
                }
                fn fields(&mut self, f: [&'b [u8]; 7]) { self.exact += 1; self.dec.event_exact(&f, self.out); }
            }
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            let s = wire::records::<7, _>(body, S { dec: &dec, out: &mut out, slow: 0, exact: 0 });
            (slow, exact) = (s.slow, s.exact);
            std::hint::black_box(&out.ev);
        });
        eprintln!("fallback: shape {slow} ({:.3}%), poisoned/simple {exact}", 100.0 * slow as f64 / rows_n as f64);
        // BENCH_ALL=1: the fallback rate over every event file of the dataset.
        if std::env::var_os("BENCH_ALL").is_some() {
            struct S<'a, 'c, 'o> { dec: &'a Decoder<'a, 'c>, out: &'o mut Scan, rows: usize, slow: usize, exact: usize }
            impl<'b> wire::Sink<'b, 7> for S<'_, '_, '_> {
                #[inline(always)]
                fn record(&mut self, f: wire::Fields<'b, 7>) {
                    self.rows += 1;
                    if !self.dec.fast(&f, self.out) { self.slow += 1; self.dec.event_exact(&f.all(), self.out); }
                }
                fn fields(&mut self, f: [&'b [u8]; 7]) { self.rows += 1; self.exact += 1; self.dec.event_exact(&f, self.out); }
            }
            let (mut rows, mut slow, mut exact) = (0, 0, 0);
            for p in list_csv(&data.join("events")).unwrap() {
                let buf = fs::read(&p).unwrap();
                let mut out = Scan { ev: bins(1 << b.user), cl: Vec::new(), cand: bins(1 << b.id), wide: Vec::new(), links: Vec::new() };
                let s = wire::records::<7, _>(crate::body(&buf), S { dec: &dec, out: &mut out, rows: 0, slow: 0, exact: 0 });
                rows += s.rows; slow += s.slow; exact += s.exact;
            }
            eprintln!("all files: rows {rows} fallback shape {slow} ({:.3}%) poisoned/simple {exact} ({:.3}%)",
                      100.0 * slow as f64 / rows as f64, 100.0 * exact as f64 / rows as f64);
        }
        eprintln!("rows={rows_n} bytes={}", body.len());
    }
}
