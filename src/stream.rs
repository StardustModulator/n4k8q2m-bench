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

/// One multiply by an odd constant: a bijection whose top bits depend on
/// every bit of the key, while its low bits depend only on low key bits.
/// Event ids put their entropy in the id's bytes, so the top bits of the
/// product pick partitions and filter slots well; anything indexing by other
/// bits mixes further first.
#[inline(always)]
fn mix(x: u64) -> u64 {
    x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
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
    h: u64,   // mix(event key): top bits pick the partition
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
        let h = mix(id);
        let part = (h >> (64 - b.id)) as usize;
        let loc = match at_row {
            None => DEAD as u64,
            Some((bin, row)) => (row as u64) << b.user | bin as u64,
        };
        if (loc < DEAD as u64 || at_row.is_none()) && (0..=u32::MAX as i64).contains(&sec) {
            self.cand[part].push(Cand { h, sec: sec as u32, loc: loc as u32 });
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

/// Rows per decode batch. The scanner's fast path only stores a record's
/// offsets (16 bytes), so it stays small enough to inline at every call
/// site, and each decoder runs as one loop with its constants in registers.
const BATCH: usize = 32;

/// A fast-path record: its offset in the window and its separator offsets.
#[derive(Clone, Copy)]
struct Rec<const N: usize> {
    row: u16,
    ends: [u16; N],
}

/// A row decoder. A trait rather than a closure so the implementation can
/// demand inlining into the batch loop, where its constants stay in registers.
trait Decode<'b, const N: usize> {
    fn row(&mut self, f: &[&'b [u8]; N]);
}

impl<'b, const N: usize, D: Decode<'b, N>> Decode<'b, N> for &mut D {
    #[inline(always)]
    fn row(&mut self, f: &[&'b [u8]; N]) {
        (**self).row(f);
    }
}

/// Gathers fast-path records of one window into a batch and decodes them in
/// bulk; slow-path records are decoded as they come. Rows are independent,
/// so the interleaving does not matter.
struct Batch<'b, const N: usize, D: Decode<'b, N>> {
    decode: D,
    base: *const u8,
    n: usize,
    recs: [Rec<N>; BATCH],
    _buf: std::marker::PhantomData<&'b [u8]>,
}

impl<'b, const N: usize, D: Decode<'b, N>> Batch<'b, N, D> {
    fn new(decode: D) -> Self {
        Batch { decode, base: std::ptr::null(), n: 0, recs: [Rec { row: 0, ends: [0; N] }; BATCH], _buf: std::marker::PhantomData }
    }

    #[inline(never)]
    fn flush(&mut self) {
        for r in &self.recs[..self.n] {
            // SAFETY: every record was handed over by the scanner for the
            // window at `base` (see `wire::Sink::fast`).
            let f: [&'b [u8]; N] = unsafe { wire::unquoted(self.base, r.row as usize, &r.ends) };
            self.decode.row(&f);
        }
        self.n = 0;
    }

    #[inline(never)]
    fn slow_row(&mut self, f: [&'b [u8]; N]) {
        self.decode.row(&f);
    }
}

impl<'b, const N: usize, D: Decode<'b, N>> wire::Sink<'b, N> for Batch<'b, N, D> {
    #[inline(always)]
    fn fast(&mut self, base: *const u8, row: usize, ends: &[u16; N]) {
        if self.n == BATCH || base != self.base {
            self.flush();
            self.base = base;
        }
        self.recs[self.n] = Rec { row: row as u16, ends: *ends };
        self.n += 1;
    }
    #[inline(always)]
    fn slow(&mut self, f: [&'b [u8]; N]) {
        self.slow_row(f);
    }
    fn finish(&mut self) {
        self.flush();
    }
}

#[inline(always)]
fn batched<'b, const N: usize>(buf: &'b [u8], decode: impl Decode<'b, N>) {
    wire::records::<N>(buf, wire::WINDOW, Batch::new(decode));
}

struct EventRows<'a, 'c, 'd> {
    dec: &'a Decoder<'c, 'd>,
    out: &'a mut Scan,
}

impl<'b> Decode<'b, 7> for EventRows<'_, '_, '_> {
    #[inline(always)]
    fn row(&mut self, f: &[&'b [u8]; 7]) {
        self.dec.event(f, self.out);
    }
}

struct ClickRows<'a, 'b, 'c, 'd> {
    dec: &'a Decoder<'c, 'd>,
    out: &'a mut Scan,
    buf: &'b [u8],
}

impl<'b> Decode<'b, 4> for ClickRows<'_, 'b, '_, '_> {
    #[inline(always)]
    fn row(&mut self, f: &[&'b [u8]; 4]) {
        self.dec.click(self.buf, f, self.out);
    }
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
            v.push(Event { key, val });
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
        out.cl[(key >> (64 - self.b.user)) as usize].push(Click { key, hit });
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
    // Slot value: part index + 1 in the high half, row in the low half. Only
    // suspects reach the table, so mixing `h` once more here is cheap and
    // makes the slot independent of the prefix bits used above.
    let mask = (suspects * 2).next_power_of_two() - 1;
    let mut table = vec![0u64; mask + 1];
    for (p, part) in parts.iter().enumerate() {
        for (i, c) in part.iter().enumerate() {
            let (w, bit) = probe(c.h);
            if marks[w][1] & bit == 0 { continue; }
            let me = ((p as u64 + 1) << 32) | i as u64;
            let mut s = hash(c.h) as usize & mask;
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

/// Minor faults and user+system CPU ticks of the process so far (Linux only).
fn rusage()->(u64,u64){
    let s=fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let f:Vec<&str>=s.rsplit(')').next().unwrap_or("").split_whitespace().collect();
    let g=|i:usize|f.get(i).and_then(|x|x.parse::<u64>().ok()).unwrap_or(0);
    (g(7),g(11)+g(12))
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
    let env_pow2=|k:&str,d:usize|std::env::var(k).ok().and_then(|s|s.parse::<usize>().ok()).map_or(d,|v|v.next_power_of_two().clamp(2,1<<16));
    // Few write streams: all bin headers and tails must stay in L1 while a
    // row is decoded (x86 cores have 32-48 KiB). Dedup partitions are read
    // back whole, so each should still fit L2: about 64 K candidates.
    let eb=env_pow2("SS_EB",(event_bytes / (77 * 65536)).next_power_of_two().clamp(8,512));
    let pb=env_pow2("SS_PB",(event_bytes / (77 * 131072)).next_power_of_two().clamp(16,256));
    let abl=std::env::var("ABL").ok().and_then(|s|s.parse::<u8>().ok()).unwrap_or(0);
    let b=Bits{user:pb.trailing_zeros(),id:eb.trailing_zeros()};
    chunks.sort_unstable_by_key(|c|std::cmp::Reverse(c.end-c.start));
    let next=AtomicUsize::new(0);
    let shared=Shared{slow:Mutex::new(SlowIntern::new())};
    let dec=Decoder::new(&cx,&clock,&shared,origin,b);
    let prep=total.elapsed();
    let r0=rusage();
    // Workers take chunks dynamically; leave headroom over an even share.
    let share=|bytes:usize,row:usize,n:usize|bytes/row/threads*5/4/n+64;
    let scans=thread::scope(|scope|{
        let handles:Vec<_>=(0..threads).map(|_|scope.spawn(||{
            let mut out=Scan{ev:sized(pb,share(event_bytes,77,pb)),cl:sized(pb,share(bytes[1],47,pb)),
                cand:sized(eb,share(event_bytes,77,eb)),wide:Vec::new(),links:Vec::new()};
            let mut acc=0u64;
            loop {
                let i=next.fetch_add(1,AtomicOrdering::Relaxed);
                let Some(c)=chunks.get(i) else{break};
                let buf=&maps[c.file as usize][c.start..c.end];
                match (c.kind,abl) {
                    (0,1)=>wire::rows::<7>(buf,|f|acc+=(f[0].len()+f[6].len()) as u64),
                    (1,1)=>wire::rows::<4>(buf,|f|acc+=(f[0].len()+f[3].len()) as u64),
                    (0,_)=>batched::<7>(buf,EventRows{dec:&dec,out:&mut out}),
                    (1,_)=>batched::<4>(buf,ClickRows{dec:&dec,out:&mut out,buf}),
                    _=>wire::rows::<2>(buf,|[a,b]| {
                        if !a.is_empty() && !b.is_empty() && a!=b {
                            out.links.push((user_key(&shared,a),user_key(&shared,b)));
                        }
                    }),
                }
            }
            std::hint::black_box(acc);
            out
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).collect::<Vec<_>>()
    });
    let parsed=total.elapsed();
    let r1=rusage();
    if abl!=0 {
        eprintln!("ms prep={} parse={} abl={} flt={} cpu={}",prep.as_millis(),(parsed-prep).as_millis(),abl,r1.0-r0.0,(r1.1-r0.1)*10);
        std::process::exit(0);
    }
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
    let aliases=links::build(lks,&slow,threads,pb);
    let indexed=total.elapsed();
    let rows=Rows::new(&mut evs,b.user);
    consume(cands,threads,||(),|_,(c,wd)|dedup(c,wd,&rows,b.id));
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
        let r2=rusage();
        eprintln!("ms prep={} parse={} links={} dedup={} rename={} group={} merge={} pflt={} pcpu={} rflt={} rcpu={}",
            prep.as_millis(),(parsed-prep).as_millis(),(indexed-parsed).as_millis(),(reduced-indexed).as_millis(),
            (renamed-reduced).as_millis(),(grouped-renamed).as_millis(),(total.elapsed()-grouped).as_millis(),
            r1.0-r0.0,(r1.1-r0.1)*10,r2.0-r1.0,(r2.1-r1.1)*10);
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
            struct Len(usize);
            impl<'b> Decode<'b, 7> for Len {
                #[inline(always)]
                fn row(&mut self, f: &[&'b [u8]; 7]) { self.0 += f[0].len() + f[6].len(); }
            }
            let mut acc = Len(0);
            wire::records::<7>(body, wire::WINDOW, Batch::new(&mut acc));
            std::hint::black_box(acc.0);
        });
        // Per-field decoder costs over pre-extracted fields: includes each
        // field's own branch behaviour but no scanning.
        let mut rows_v: Vec<[&[u8]; 7]> = Vec::with_capacity(rows_n);
        wire::rows::<7>(body, |f| rows_v.push(f));
        let per = |name: &str, f: &mut dyn FnMut(&[&[u8]; 7]) -> u64| {
            if only.as_deref().is_some_and(|o| o != name) { return; }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let mut acc = 0u64;
                for r in &rows_v { acc ^= f(r); }
                std::hint::black_box(acc);
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!("{name:>12}: {:6.2} ns/row (fields only)", best * 1e9 / rows_v.len() as f64);
        };
        per("f:loop", &mut |f| f[0].len() as u64);
        per("f:event_key", &mut |f| event_key(&shared, f[0]).unwrap_or(0));
        per("f:user_key", &mut |f| user_key(&shared, f[1]));
        per("f:ts", &mut |f| clock.parse(f[2]).unwrap_or(0) as u64);
        per("f:ingest", &mut |f| clock.ingest_s(f[3]).unwrap_or(0) as u64);
        per("f:payload", &mut |f| dec.payload(f, 0).unwrap_or(0));
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
        report("scan_events", &mut || {
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            batched::<7>(body, EventRows { dec: &dec, out: &mut out });
            std::hint::black_box(&out.ev);
        });
        eprintln!("rows={rows_n} bytes={}", body.len());
    }
}
