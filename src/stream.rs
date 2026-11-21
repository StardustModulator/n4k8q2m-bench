//! Each event row is written once, straight into its raw user's bin. Dedup
//! sees only a 16-byte candidate per row, partitioned by event ID, and kills
//! losing copies in place. Renaming then compacts each user bin and scatters
//! only renamed rows; neither an all-user interner nor a global sort is needed.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::{histories, Click, Event, Histories};
use crate::clock::Clock;
use crate::links::Aliases;

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
            debug_assert!(part < self.cand.len());
            // SAFETY: part < 1 << id == self.cand.len(), asserted per batch.
            unsafe { self.cand.get_unchecked_mut(part) }.push(Cand { h, sec: sec as u32, loc: loc as u32 });
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

/// A payload word that is never valid: its price would exceed 10,000,000.
const NONE: u64 = u64::MAX;

/// Rows per decode batch. The scanner invokes its callback from several
/// sites; a callback that only stores fields stays small enough to inline,
/// and each decoder runs as one loop with its constants in registers.
const BATCH: usize = 128;
const _: () = assert!(BATCH <= 256); // stamps() lists rows as u8

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

    /// Column `col` of a batch as timestamps, i64::MIN when invalid (no
    /// valid instant is that early). Rows are grouped by form first: in
    /// shuffled input a per-row form branch mispredicts about every other row.
    #[inline(never)]
    fn stamps<const N: usize>(&self, rows: &[[&[u8]; N]], col: usize) -> [i64; BATCH] {
        debug_assert!(rows.len() <= BATCH);
        // Every slot is written below: each row joins exactly one form.
        let mut ts = [0i64; BATCH];
        let mut form = [[0u8; BATCH]; 3];
        let (mut a, mut b, mut c) = (0, 0, 0);
        for (i, f) in rows.iter().enumerate() {
            let n = f[col].len();
            let (f1, f2) = (n == 20, n == 25);
            // SAFETY: each cursor counts earlier rows, so it is < i + 1 <= BATCH.
            unsafe {
                *form[0].get_unchecked_mut(a) = i as u8;
                *form[1].get_unchecked_mut(b) = i as u8;
                *form[2].get_unchecked_mut(c) = i as u8;
            }
            a += usize::from(f1);
            b += usize::from(f2);
            c += usize::from(!(f1 | f2));
        }
        // SAFETY (all three loops): listed indexes are rows of this batch, and
        // a row is listed under F1 or F2 only when its field has that length.
        let field = |i: u8| unsafe { *rows.get_unchecked(usize::from(i)).get_unchecked(col) };
        for &i in &form[0][..a] {
            let t = self.clock.parse20(unsafe { &*field(i).as_ptr().cast() });
            unsafe { *ts.get_unchecked_mut(usize::from(i)) = t.unwrap_or(i64::MIN) };
        }
        for &i in &form[1][..b] {
            let t = self.clock.parse25(unsafe { &*field(i).as_ptr().cast() });
            unsafe { *ts.get_unchecked_mut(usize::from(i)) = t.unwrap_or(i64::MIN) };
        }
        for &i in &form[2][..c] {
            let t = self.clock.parse(field(i));
            unsafe { *ts.get_unchecked_mut(usize::from(i)) = t.unwrap_or(i64::MIN) };
        }
        ts
    }

    /// One column pass at a time, each its own function: a pass keeps its
    /// constants in registers, where one fused row loop spilled and
    /// rematerialized them on every row.
    #[inline(never)]
    fn events(&self, rows: &[[&[u8]; 7]], out: &mut Scan) {
        assert!(out.ev.len() == 1 << self.b.user && out.cand.len() == 1 << self.b.id);
        let ts = self.stamps(rows, 2);
        let sec = self.ingests(rows);
        let val = self.payloads(rows, &ts, &sec);
        let key = self.users(rows, &val);
        for (i, f) in rows.iter().enumerate() {
            if sec[i] == i64::MIN { continue; }
            let Some(id) = event_key(self.s, f[0]) else { continue };
            let mut at_row = None;
            if val[i] != NONE {
                let bin = (key[i] >> (64 - self.b.user)) as usize;
                // SAFETY: bin < 1 << user == out.ev.len(), asserted above.
                let v = unsafe { out.ev.get_unchecked_mut(bin) };
                at_row = Some((bin, v.len()));
                v.push(Event { key: key[i], val: val[i] });
            }
            out.candidate(id, sec[i], at_row, self.b);
        }
    }

    /// Ingest seconds; i64::MIN marks a row to drop before dedup.
    #[inline(never)]
    fn ingests(&self, rows: &[[&[u8]; 7]]) -> [i64; BATCH] {
        let mut sec = [i64::MIN; BATCH];
        for (s, f) in sec.iter_mut().zip(rows) {
            *s = self.clock.ingest_s(f[3]).unwrap_or(i64::MIN);
        }
        sec
    }

    #[inline(never)]
    fn payloads(&self, rows: &[[&[u8]; 7]], ts: &[i64; BATCH], sec: &[i64; BATCH]) -> [u64; BATCH] {
        let mut val = [NONE; BATCH];
        for (i, f) in rows.iter().enumerate() {
            if sec[i] != i64::MIN {
                val[i] = self.payload(f, ts[i], sec[i] * 1000).unwrap_or(NONE);
            }
        }
        val
    }

    /// Mixed user keys of the rows with a payload.
    #[inline(never)]
    fn users(&self, rows: &[[&[u8]; 7]], val: &[u64; BATCH]) -> [u64; BATCH] {
        let mut key = [0u64; BATCH];
        for (i, f) in rows.iter().enumerate() {
            if val[i] != NONE {
                key[i] = hash(user_key(self.s, f[1]));
            }
        }
        key
    }

    #[inline(always)]
    fn payload(&self, f: &[&[u8]; 7], ts: i64, at: i64) -> Option<u64> {
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
        // One unsigned compare per bound pair; rejects the invalid sentinel.
        if ts.wrapping_sub(self.start) as u64 >= self.span || at.wrapping_sub(ts) > DAY_MS {
            return None;
        }
        let pay = if purchase { parse_price(f[6])? | u32::from(product_cat(self.cx, f[5])) << 24 } else { 0 };
        Some(((ts - self.origin) as u64) << 32 | u64::from(pay))
    }

    #[inline(never)]
    fn clicks(&self, buf: &[u8], rows: &[[&[u8]; 4]], out: &mut Scan) {
        assert!(out.cl.len() == 1 << self.b.user);
        let ts = self.stamps(rows, 3);
        for (f, &ts) in rows.iter().zip(&ts) {
            self.click(buf, f, ts, out);
        }
    }

    // All copies of a click are identical. Window membership is existential
    // per campaign and credit is a min/max over keys. Both are idempotent:
    // duplicating an identical tuple cannot change either, so clicks skip dedup.
    #[inline(always)]
    fn click(&self, buf: &[u8], f: &[&[u8]; 4], ts: i64, out: &mut Scan) {
        if f[0].is_empty() || f[1].is_empty() { return; }
        if ts.wrapping_sub(self.click_start) as u64 >= self.click_span { return; }
        let Some(camp) = self.campaign(buf, f[2]) else { return };
        let key = hash(user_key(self.s, f[1]));
        let hit = Hit { ts: (ts - self.origin) as u32, id: user_key(self.s, f[0]), camp };
        // SAFETY: the bin is < 1 << user == out.cl.len(), asserted per batch.
        unsafe { out.cl.get_unchecked_mut((key >> (64 - self.b.user)) as usize) }.push(Click { key, hit });
    }

    #[inline(never)]
    fn links(&self, rows: &[[&[u8]; 2]], out: &mut Scan) {
        for &[a, b] in rows {
            if !a.is_empty() && !b.is_empty() && a != b {
                out.links.push((user_key(self.s, a), user_key(self.s, b)));
            }
        }
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
    // Suspects are gathered without a branch (about one row in ten is one),
    // then contested in a table that holds each id's current winner inline.
    let mask = (suspects * 2).next_power_of_two() - 1;
    let mut table = vec![Winner::default(); mask + 1];
    let mut list = Vec::<u32>::with_capacity(suspects + 1);
    for (p, part) in parts.iter().enumerate() {
        list.clear();
        let lp = list.as_mut_ptr();
        let mut k = 0;
        for (i, c) in part.iter().enumerate() {
            let (w, bit) = probe(c.h);
            // SAFETY: k counts this part's suspects so far, and all parts
            // together hold exactly `suspects`, so k < capacity.
            unsafe { *lp.add(k) = i as u32 };
            k += usize::from(marks[w][1] & bit != 0);
        }
        // SAFETY: the first k entries were written above.
        unsafe { list.set_len(k) };
        for &i in &list {
            let c = part[i as usize];
            let mut s = c.h as usize & mask;
            loop {
                let t = &mut table[s];
                if t.h == 0 {
                    *t = Winner { h: c.h, sec: c.sec, loc: c.loc, part: p as u32 };
                    break;
                }
                if t.h == c.h {
                    // Copies with equal ingest are identical: keep either.
                    if c.sec < t.sec {
                        rows.kill(t.part as usize, t.loc);
                        *t = Winner { h: c.h, sec: c.sec, loc: c.loc, part: p as u32 };
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

/// The earliest copy of an id so far; h is never 0, so 0 marks a free slot.
#[derive(Clone, Copy, Default)]
struct Winner {
    h: u64,
    sec: u32,
    loc: u32,
    part: u32,
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
    slots: Vec<u32>, // occupied slots, so a reset clears only those
}

impl Names {
    fn reset(&mut self, rows: usize) {
        // A raw user recurs about a dozen times per bin: start near 25% load.
        let n = (rows / 4).max(64).next_power_of_two();
        if n == self.table.len() {
            for &i in &self.slots { self.table[i as usize] = (0, 0, 0); }
        } else {
            self.table.clear();
            self.table.resize(n, (0, 0, 0));
        }
        self.slots.clear();
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
        self.slots.push(i as u32);
        if self.slots.len() * 2 > self.table.len() {
            let grown = vec![(0, 0, 0); self.table.len() * 4];
            let old = std::mem::replace(&mut self.table, grown);
            let mask = self.table.len() - 1;
            self.slots.clear();
            for e in old.into_iter().filter(|e| e.0 != 0) {
                let mut j = e.0 as usize & mask;
                while self.table[j].0 != 0 { j = (j + 1) & mask; }
                self.table[j] = e;
                self.slots.push(j as u32);
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

/// Concatenate the workers' user rows, one parallel copy per worker (a serial
/// append moves hundreds of megabytes on one core), and sum their campaign stats.
fn gather(mut partials:Vec<Acc>,cells:usize)->(Vec<UserRow>,Vec<[i64;4]>) {
    let mut stats=vec![[0i64;4];cells];
    for p in &partials {
        for (t,v) in stats.iter_mut().zip(&p.stats) {
            for i in 0..4{t[i]+=v[i];}
        }
    }
    if partials.len()==1 {return (partials.pop().unwrap().rows,stats);}
    let n=partials.iter().map(|p|p.rows.len()).sum();
    let mut users=Vec::<UserRow>::with_capacity(n);
    thread::scope(|scope|{
        let mut rest=&mut users.spare_capacity_mut()[..n];
        for p in &mut partials {
            let (dst,tail)=std::mem::take(&mut rest).split_at_mut(p.rows.len());
            rest=tail;
            let src=&mut p.rows;
            scope.spawn(move||{
                // SAFETY: dst and src have equal lengths and do not overlap.
                // The rows move bitwise; emptying src without dropping them
                // leaves dst their only owner.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(),dst.as_mut_ptr().cast::<UserRow>(),src.len());
                    src.set_len(0);
                }
            });
        }
    });
    // SAFETY: the copies above initialized all n rows.
    unsafe {users.set_len(n)};
    (users,stats)
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
            chunks.extend((0..m.len().div_ceil(CHUNK)).map(|k|ChunkLite{file:maps.len(),kind,k}));
            maps.push(m);
        }
    }
    let event_bytes=bytes[0];
    let threads=threads.max(1).min(chunks.len().max(1));
    let eb=(event_bytes / (77 * 16384)).next_power_of_two().clamp(8,2048);
    let pb=(event_bytes / (77 * 65536)).next_power_of_two().clamp(16,1024);
    let b=Bits{user:pb.trailing_zeros(),id:eb.trailing_zeros()};
    chunks.sort_unstable_by_key(|c|std::cmp::Reverse(CHUNK.min(maps[c.file].len()-c.k*CHUNK)));
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
                let m=&maps[c.file];
                // One call maps the chunk's resident pages instead of a fault
                // per few pages. Advisory only: older kernels refuse it.
                #[cfg(target_os="linux")]
                let _=m.advise_range(Advice::PopulateRead,c.k*CHUNK,CHUNK.min(m.len()-c.k*CHUNK));
                let (start,end)=chunk_bounds(m,c.k);
                let buf=&m[start..end];
                match c.kind {
                    0=>batched::<7>(buf,|rows|dec.events(rows,&mut out)),
                    1=>batched::<4>(buf,|rows|dec.clicks(buf,rows,&mut out)),
                    _=>batched::<2>(buf,|rows|dec.links(rows,&mut out)),
                }
            }
            out
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).collect::<Vec<_>>()
    });
    let parsed=total.elapsed();
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
    // Linking and dedup are independent. Running them side by side lets
    // dedup jobs fill the cores that the link build's serial steps leave idle.
    let rows=Rows::new(&mut evs,b.user);
    let (aliases,indexed)=thread::scope(|scope|{
        let linker=scope.spawn(||{let a=links::build(lks,&slow,threads,pb);(a,total.elapsed())});
        consume(cands,threads,||(),|_,(c,wd)|dedup(c,wd,&rows,b.id));
        linker.join().unwrap()
    });
    drop(rows);
    let reduced=total.elapsed();
    let shift=64-b.user;
    // Renamed rows land in every bin; reserve near half of an even share so
    // the scatter rarely regrows (untouched capacity costs no pages).
    let nev=evs.iter().flatten().map(Vec::len).sum::<usize>()/pb/threads*3/4;
    let ncl=cls.iter().flatten().map(Vec::len).sum::<usize>()/pb/threads*3/4;
    let hist=||(0..pb).map(|_| Histories{ev:Vec::with_capacity(nev),cl:Vec::with_capacity(ncl)}).collect::<Vec<_>>();
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
    let (users,stats)=gather(partials,camps.len()*days);
    let mut camps_out=Vec::new();
    for (i,[sessions,purchasing,revenue,assisted]) in stats.into_iter().enumerate(){
        if sessions!=0 || assisted!=0{
            camps_out.push(CampRow{camp:(i/days) as u16,day:w.start_day+(i%days) as i32,sessions,purchasing,revenue,assisted});
        }
    }
    if std::env::var_os("STAGES").is_some() {
        eprintln!("ms prep={} parse={} links={} dedup={} rename={} group={} merge={}",
            prep.as_millis(),(parsed-prep).as_millis(),(indexed-parsed).as_millis(),(reduced-indexed).as_millis(),
            (renamed-reduced).as_millis(),(grouped-renamed).as_millis(),(total.elapsed()-grouped).as_millis());
    }
    // The process exits right after writing; unmapping is wasted time.
    std::mem::forget(maps);
    std::mem::forget(aliases);
    Ok(Tables{users,camps_out,camps,cat_names,slow,organic})
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

    /// Every copy contest, against a reference minimum over (ingest, order).
    #[test]
    fn dedup_keeps_the_earliest_copy() {
        let bits = 2; // four user bins per worker
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for round in 0..200 {
            let workers = 1 + round % 3;
            let ids = 1 + (next() % 40) as u64;
            let mut evs: Vec<Vec<Vec<Event>>> = (0..workers).map(|_| bins(1 << bits)).collect();
            let mut parts: Vec<Vec<Cand>> = (0..workers).map(|_| Vec::new()).collect();
            let mut wide: Vec<Vec<Wide>> = (0..workers).map(|_| Vec::new()).collect();
            // (id, ingest, worker, bin, row) of every copy that has a payload.
            let mut copies = Vec::new();
            let mut all = Vec::new();
            for _ in 0..next() % 120 {
                let id = 1 + next() % ids;
                let w = (next() % workers as u64) as usize;
                let sec = (next() % 6) as u32;
                let valid = next() % 4 != 0;
                let h = hash(id);
                let at_row = valid.then(|| {
                    let bin = (next() % 4) as usize;
                    let row = evs[w][bin].len();
                    evs[w][bin].push(Event { key: 1 + next() % 9, val: 0 });
                    copies.push((id, sec, w, bin, row));
                    (bin, row)
                });
                all.push((id, sec, valid, w, at_row));
                if round % 2 == 1 && next() % 8 == 0 {
                    wide[w].push(Wide { h, at: i64::from(sec) * 1000, at_row });
                } else {
                    let loc = at_row.map_or(DEAD, |(bin, row)| (row << bits | bin) as u32);
                    parts[w].push(Cand { h, sec, loc });
                }
            }
            // Reference: for each id the minimum ingest wins. Real ties are
            // identical copies; a random tie of valid and invalid may go either way.
            let rows = Rows::new(&mut evs, bits);
            dedup(parts, wide, &rows, 0);
            drop(rows);
            let alive = |w: usize, bin: usize, row: usize| evs[w][bin][row].key != 0;
            for id in 1..=ids {
                let mine: Vec<_> = all.iter().filter(|c| c.0 == id).collect();
                if mine.is_empty() { continue; }
                let best = mine.iter().map(|c| c.1).min().unwrap();
                let survivors: Vec<_> = copies.iter().filter(|c| c.0 == id && alive(c.2, c.3, c.4)).collect();
                let earliest_valid = mine.iter().any(|c| c.1 == best && c.2);
                let earliest_invalid = mine.iter().any(|c| c.1 == best && !c.2);
                assert!(survivors.len() <= 1, "round {round} id {id}");
                if let Some(s) = survivors.first() { assert_eq!(s.1, best); }
                if !earliest_invalid { assert_eq!(survivors.len(), usize::from(earliest_valid), "round {round} id {id}"); }
                if !earliest_valid { assert!(survivors.is_empty(), "round {round} id {id}"); }
            }
        }
    }

    #[test]
    fn chunks_tile_the_body_at_line_starts() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for round in 0..24 {
            let mut b = b"header,line\n".to_vec();
            let target = [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 17][round % 6] + (next() % 3000) as usize;
            while b.len() < target {
                // Mostly short lines, sometimes one longer than a chunk.
                let n = if next() % 500 == 0 { CHUNK + (next() % 5000) as usize } else { (next() % 120) as usize };
                b.extend(std::iter::repeat_n(b'x', n));
                b.push(b'\n');
            }
            if round % 2 == 1 { b.extend_from_slice(b"no newline at end"); }
            let header = memchr::memchr(b'\n', &b).unwrap() + 1;
            let mut at = header;
            for k in 0..b.len().div_ceil(CHUNK) {
                let (start, end) = chunk_bounds(&b, k);
                if start == end { continue; }
                assert_eq!(start, at, "round {round} chunk {k}");
                assert!(start == 0 || b[start - 1] == b'\n');
                at = end;
            }
            assert_eq!(at, b.len(), "round {round}");
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
                let v = dec.payload(&f, clock.parse(f[2]).unwrap_or(i64::MIN), at).unwrap_or(0);
                acc ^= id ^ at as u64 ^ v;
            });
            std::hint::black_box(acc);
        });
        report("+user", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| {
                let Some(id) = id_key(&shared, f[0]) else { return };
                let Some(at) = clock.ingest(f[3]) else { return };
                let v = dec.payload(&f, clock.parse(f[2]).unwrap_or(i64::MIN), at).unwrap_or(0);
                let u = if f[1].is_empty() { 0 } else { user_key(&shared, f[1]) };
                acc ^= id ^ at as u64 ^ v ^ u;
            });
            std::hint::black_box(acc);
        });
        report("scan_events", &mut || {
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            batched::<7>(body, |rows| dec.events(rows, &mut out));
            std::hint::black_box(&out.ev);
        });
        eprintln!("rows={rows_n} bytes={}", body.len());
        let cbuf = fs::read(data.join("clicks/part-00006.csv")).unwrap();
        let cbody = &cbuf[memchr::memchr(b'\n', &cbuf).unwrap() + 1..];
        let crows = memchr::memchr_iter(b'\n', cbody).count();
        let creport = |name: &str, f: &mut dyn FnMut()| {
            if only.as_deref().is_some_and(|o| o != name) { return; }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!("{name:>12}: {:6.2} ns/row {:6.2} GB/s", best * 1e9 / crows as f64, cbody.len() as f64 / best / 1e9);
        };
        creport("c:rows<4>", &mut || {
            let mut acc = 0usize;
            wire::rows::<4>(cbody, |f| acc += f[0].len() + f[1].len() + f[2].len() + f[3].len());
            std::hint::black_box(acc);
        });
        creport("c:batch", &mut || {
            let mut acc = 0usize;
            batched::<4>(cbody, |rows| { for f in rows { acc += f[0].len() + f[1].len() + f[2].len() + f[3].len(); } });
            std::hint::black_box(acc);
        });
        creport("c:scan", &mut || {
            let mut out = Scan { ev: Vec::new(), cl: sized(1 << b.user, crows >> b.user << 1),
                                 cand: Vec::new(), wide: Vec::new(), links: Vec::new() };
            batched::<4>(cbody, |rows| dec.clicks(cbody, rows, &mut out));
            std::hint::black_box(&out.cl);
        });
        eprintln!("click rows={crows} bytes={}", cbody.len());
    }
}
