//! Each event row is written once, straight into its raw user's bin. Dedup
//! sees only a 16-byte candidate per row, partitioned by event ID, and kills
//! losing copies in place. Renaming then compacts each user bin and scatters
//! only renamed rows; neither an all-user interner nor a global sort is needed.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::{histories, Click, Event, Histories};
use crate::clock::Clock;
use crate::links::Aliases;
use crate::wire::{Lists, POISON, WINDOW};

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
const LOW7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const ZEROS: u64 = 0x3030_3030_3030_3030;

#[cfg(test)]
static SLOW_ROWS: AtomicUsize = AtomicUsize::new(0);

/// Unaligned little-endian word at `p`.
///
/// # Safety
/// Eight bytes at `p` must be readable.
#[inline(always)]
unsafe fn ld8(p: *const u8) -> u64 {
    // SAFETY: forwarded.
    u64::from_le(unsafe { p.cast::<u64>().read_unaligned() })
}

/// Bytes 0..=6 of `rel` strictly increasing; byte 7 is ignored.
#[inline(always)]
fn ascending7(rel: u64) -> bool {
    // Byte i of d is 0x80 + r[i] - r[i-1] (byte 0: 0x80 + r[0]): no borrow
    // crosses a byte, since every minuend has its high bit set and every
    // subtrahend is below 0x80. Strict increase means bytes 1..=6 are all at
    // least 0x81: high bit set and low seven bits nonzero.
    let d = (rel | HIGH).wrapping_sub((rel << 8) & LOW7);
    let nz = (d & LOW7).wrapping_add(LOW7) & HIGH;
    const BYTES: u64 = 0x0080_8080_8080_8000;
    (d & nz) & BYTES == BYTES
}

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

    #[inline(never)]
    fn events(&self, rows: &[[&[u8]; 7]], out: &mut Scan) {
        for f in rows {
            self.event(f, out);
        }
    }

    /// The event rows of one chunk, which starts at a record boundary. Stage 1
    /// lists each window's separators; then every record is certified from
    /// seven stop bytes and decoded in one fixed sequence, or handed to the
    /// exact row path: one that fails certification, one the fixed sequence
    /// does not cover, or one too close to the end of `buf` for its loads.
    #[inline(never)]
    fn event_rows(&self, buf: &[u8], out: &mut Scan, l: &mut Lists) {
        let len = buf.len();
        let mut start = 0usize;
        while start < len {
            let end = (start + WINDOW).min(len);
            let (_, nl) = wire::scan_window(buf, start, end, l);
            if nl == 0 {
                // A full window without LF: one long record.
                let lf = memchr::memchr(b'\n', &buf[end..]).map_or(len, |i| end + i);
                self.slow_event(buf, start, lf, out);
                start = lf + 1;
                continue;
            }
            let stops = l.stops.as_ptr();
            let mut o = start; // the record's first byte
            let mut cur = 0usize; // its first stop
            for &line in &l.lines[..nl] {
                let lf = start + (line & !POISON) as usize;
                // SAFETY: every record's stops end at its own LF (or EOF),
                // which is listed, so cur never passes the stops written, and
                // the list has eight bytes of slack after them.
                let st = unsafe { ld8(stops.add(cur)) };
                // Field ends relative to the record: (stop - origin) mod 128
                // per byte. The seventh must be this record's LF and the
                // seven must ascend; together they prove that exactly these
                // seven separators lie in the record, which is under 128
                // bytes (an LF in the first six would repeat the seventh's
                // byte, a comma past the sixth would push the LF out).
                let rel = (st | HIGH).wrapping_sub(u64::from((o - start) as u8 & 127) * ONES) & LOW7;
                let certified = (rel >> 48) as u8 as usize == lf - o && ascending7(rel) && line & POISON == 0 && lf + 32 <= len;
                if certified && self.fast_event(buf, o, rel, out) {
                    cur += 7;
                } else {
                    cur += self.slow_event(buf, o, lf, out);
                }
                o = lf + 1;
            }
            start = o;
        }
    }

    /// One certified record in a fixed sequence: every field is loaded and
    /// checked, and the verdicts select. Returns false, having written
    /// nothing, for a shape the sequence does not cover (an id outside the
    /// literal form, an uncached date, a rare timestamp form); those take
    /// the exact row path, which decides them as before.
    #[inline(always)]
    fn fast_event(&self, buf: &[u8], o: usize, rel: u64, out: &mut Scan) -> bool {
        // SAFETY: the record spans buf[o..o + e6] with e6 < 128 and 32 more
        // bytes of `buf` follow it; every load below starts inside the record
        // and reads at most 32 bytes from there.
        unsafe {
            let row = buf.as_ptr().add(o);
            let e = |i: u32| (rel >> (8 * i)) as u8 as usize;
            // A field is quoted exactly when its first byte is a quote: stage
            // 1 certified the record's quotes well formed.
            let q = |at: usize| usize::from(*row.add(at) == b'"');
            // Event id: `e` and seven or eight bytes in 1..=127, as event_key.
            let q0 = q(0);
            let p0 = row.add(q0);
            let n0 = e(0).wrapping_sub(2 * q0);
            let w0 = ld8(p0.add(1)) & u64::MAX.wrapping_shr((9usize.wrapping_sub(n0) as u32).wrapping_mul(8));
            let pad = u64::from(n0 == 8) << 56;
            let id_ok = ((n0 | 1) == 9) & (*p0 == b'e') & (((w0 | pad) | (w0 | pad).wrapping_sub(ONES)) & HIGH == 0);
            let id = w0 | 1 << 63;
            // User id: try_pack's eight or nine byte shape.
            let s1 = e(0) + 1;
            let q1 = q(s1);
            let p1 = row.add(s1 + q1);
            let n1 = e(1).wrapping_sub(s1).wrapping_sub(2 * q1);
            let tail = ld8(p1.add(n1.wrapping_sub(8).min(1))).swap_bytes();
            let (user_ok, ukey) = atom::pack_tail(*p1, tail, n1);
            // Timestamp: F1, F2 and 9 to 16 digits all decoded, the width selects.
            let s2 = e(1) + 1;
            let q2 = q(s2);
            let p2 = row.add(s2 + q2);
            let n2 = e(2).wrapping_sub(s2).wrapping_sub(2 * q2);
            let (hit2, mid2) = self.clock.probe(p2);
            let (hms_ok2, sec2) = clock::hms(ld8(p2.add(11)));
            let (off_ok, off) = clock::offset(ld8(p2.add(17)));
            let zone = *p2.add(19);
            let local = mid2 + sec2;
            let f1 = (n2 == 20) & (zone == b'Z') & hit2 & hms_ok2;
            let f2 = (n2 == 25) & ((zone == b'+') | (zone == b'-')) & hit2 & hms_ok2 & off_ok;
            let shifted = if zone == b'+' { local - off } else { local + off };
            let iso = (if n2 == 25 { shifted } else { local }) * 1000;
            let (dig_ok, f3v) = clock::epoch16(ld8(p2), ld8(p2.add(n2.wrapping_sub(8).min(8))), n2);
            let f3 = (n2.wrapping_sub(9) <= 7) & dig_ok;
            let ts_ok = f1 | f2 | f3;
            let ts = if f3 { f3v } else { iso };
            // Ingest: F1 only, as ingest_s.
            let s3 = e(2) + 1;
            let q3 = q(s3);
            let p3 = row.add(s3 + q3);
            let n3 = e(3).wrapping_sub(s3).wrapping_sub(2 * q3);
            let (hit3, mid3) = self.clock.probe(p3);
            let (hms_ok3, sec3) = clock::hms(ld8(p3.add(11)));
            let ingest_ok = (n3 == 20) & (*p3.add(19) == b'Z') & hit3 & hms_ok3;
            let sec = mid3 + sec3;
            // Type: whole-word compares.
            let s4 = e(3) + 1;
            let q4 = q(s4);
            let p4 = row.add(s4 + q4);
            let n4 = e(4).wrapping_sub(s4).wrapping_sub(2 * q4);
            let w4 = ld8(p4);
            let purchase = (n4 == 8) & (w4 == PURCHASE);
            let type_ok = ((n4 == 4) & ((w4 as u32 == VIEW) | (w4 as u32 == CART))) | purchase;
            // Product: non-empty.
            let s5 = e(4) + 1;
            let q5 = q(s5);
            let n5 = e(5).wrapping_sub(s5).wrapping_sub(2 * q5);
            if !(id_ok & ingest_ok & ts_ok & (user_ok | (n1 == 0))) {
                return false;
            }
            let at = sec * 1000;
            let in_window = (ts.wrapping_sub(self.start) as u64) < self.span;
            let mut valid = type_ok & (n5 != 0) & (n1 != 0) & in_window & (at - ts <= DAY_MS);
            let mut pay = 0u32;
            if purchase & valid {
                let s6 = e(5) + 1;
                let q6 = q(s6);
                match parse_price(&buf[o + s6 + q6..o + e(6) - q6]) {
                    Some(p) => pay = p | u32::from(product_cat(self.cx, &buf[o + s5 + q5..o + e(5) - q5])) << 24,
                    None => valid = false,
                }
            }
            let mut at_row = None;
            if valid {
                let key = hash(ukey);
                let bin = (key >> (64 - self.b.user)) as usize;
                let v = &mut out.ev[bin];
                at_row = Some((bin, v.len()));
                v.push(Event { key, val: ((ts - self.origin) as u64) << 32 | u64::from(pay) });
            }
            out.candidate(id, sec, at_row, self.b);
            true
        }
    }

    /// The exact path for the record `buf[o..lf]`: the general scanner on
    /// that one record, then the scalar decoder. Returns the number of
    /// separators the record holds: its commas and its LF or EOF.
    #[cold]
    #[inline(never)]
    fn slow_event(&self, buf: &[u8], o: usize, lf: usize, out: &mut Scan) -> usize {
        #[cfg(test)]
        SLOW_ROWS.fetch_add(1, AtomicOrdering::Relaxed);
        let row = &buf[o..lf];
        wire::rows::<7>(row, |f| self.event(&f, out));
        memchr::memchr_iter(b',', row).count() + 1
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
            let mut lists=Lists::new();
            loop {
                let i=next.fetch_add(1,AtomicOrdering::Relaxed);
                let Some(c)=chunks.get(i) else{break};
                let buf=&maps[c.file as usize][c.start..c.end];
                match c.kind {
                    0=>dec.event_rows(buf,&mut out,&mut lists),
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

    /// Event rows of every form the generator writes and every dirt kind the
    /// rules name, plus shapes only the exact path handles.
    fn synth_events(next: &mut impl FnMut() -> u64, w: &Window, rows: usize) -> Vec<u8> {
        const B36: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        fn word(next: &mut impl FnMut() -> u64, prefix: u8, n: usize) -> Vec<u8> {
            let mut v = vec![prefix];
            for _ in 0..n { v.push(B36[(next() % 36) as usize]); }
            v
        }
        fn iso(t: i64, zone: &str) -> Vec<u8> {
            let (d, s) = (t.div_euclid(DAY_MS), t.rem_euclid(DAY_MS) / 1000);
            let mut date = [0u8; 10];
            format_day(d as i32, &mut date);
            format!("{}T{:02}:{:02}:{:02}{zone}", std::str::from_utf8(&date).unwrap(), s / 3600, s / 60 % 60, s % 60).into_bytes()
        }
        let mut out = Vec::new();
        for _ in 0..rows {
            let mut f: Vec<Vec<u8>> = Vec::with_capacity(7);
            f.push(match next() % 64 {
                0 => Vec::new(), 1 => word(next, b'E', 8), 2 => word(next, b'e', 7), 3 => word(next, b'e', 9),
                4 => b"x1".to_vec(), 5 => word(next, b'e', 6), 6 => b"e\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f".to_vec(),
                _ => word(next, b'e', 8),
            });
            f.push(match next() % 64 {
                0 => Vec::new(), 1 => word(next, b'U', 8), 2 => word(next, b'u', 6), 3 => word(next, b'u', 9),
                4 => b" u1".to_vec(), 5 => "u\u{e9}1234567".as_bytes().to_vec(), 6 => word(next, b'k', 7),
                7 => b"u1234567\x7f".to_vec(), _ => { let n = 7 + (next() % 2) as usize; word(next, b'u', n) },
            });
            let t = w.start - 2 * DAY_MS + (next() % (18 * DAY_MS) as u64) as i64;
            let t = t / 1000 * 1000 + if next() % 8 == 0 { (next() % 1000) as i64 } else { 0 };
            let r = next() % 100;
            f.push(if r < 50 {
                if t % 1000 == 0 { iso(t, "Z") } else { t.to_string().into_bytes() }
            } else if r < 62 {
                let off = ((next() % 14) as i64 * 60 + [0, 30, 45][(next() % 3) as usize]) * 60_000;
                let neg = next() % 2 == 0;
                if t % 1000 == 0 {
                    let mut s = iso(if neg { t - off } else { t + off }, "");
                    s.extend_from_slice(format!("{}{:02}:{:02}", if neg { '-' } else { '+' }, off / 3_600_000, off / 60_000 % 60).as_bytes());
                    s
                } else { t.to_string().into_bytes() }
            } else if r < 92 {
                t.to_string().into_bytes()
            } else {
                match r {
                    92 => Vec::new(), 93 => iso(t, "z"), 94 => format!("{t:020}").into_bytes(),
                    95 => b"2028-02-30T00:00:00Z".to_vec(), 96 => iso(t, ".000Z"), 97 => b"2028-02-29T00:00:00Z".to_vec(),
                    98 => iso(t, "+24:00"), _ => b"1.79e12".to_vec(),
                }
            });
            let late = match next() % 50 { 0 => DAY_MS, 1 => DAY_MS + 1000, 2 => 3 * DAY_MS, 3 => -5000, _ => (next() % 3600) as i64 * 1000 };
            let at = (t + late) / 1000 * 1000;
            f.push(match next() % 100 {
                0 => Vec::new(), 1 => at.to_string().into_bytes(), 2 => iso(at, "+00:00"),
                3 => iso(w.start + 400 * DAY_MS, "Z"), 4 => iso(at, "z"), _ => iso(at, "Z"),
            });
            f.push(match next() % 100 {
                0..=2 => b"purchase".to_vec(), 3 => b"Purchase".to_vec(), 4 => b"cart ".to_vec(), 5 => Vec::new(),
                6..=20 => b"cart".to_vec(), _ => b"view".to_vec(),
            });
            f.push(match next() % 200 {
                0 => Vec::new(), 1 => b"P123".to_vec(), 2 => b"p0123".to_vec(), 3 => b"p99999999".to_vec(),
                _ => format!("p{}", next() % 60_000).into_bytes(),
            });
            f.push(match next() % 20 {
                0 => Vec::new(), 1 => b"0".to_vec(), 2 => b"-5".to_vec(), 3 => b"1.5".to_vec(), 4 => b"10000001".to_vec(),
                5 => b"0250".to_vec(), _ => (1 + next() % 50_000).to_string().into_bytes(),
            });
            let bad = next() % 400;
            for (i, fld) in f.iter_mut().enumerate() {
                if next() % 5 == 0 {
                    let mut q = vec![b'"'];
                    q.extend_from_slice(fld);
                    q.push(b'"');
                    *fld = q;
                }
                if bad == 1 && i == 2 { let at = fld.len().min(1); fld.insert(at, b'"'); }
            }
            match bad {
                2 => { f.pop(); }
                3 => f.push(b"extra".to_vec()),
                4 => f.clear(),
                5 => f[5] = vec![b'x'; 140],
                6 => f[6].push(b'\r'),
                7 => f[0] = b"\"".to_vec(),
                8 => f[3] = b"\"\"".to_vec(),
                _ => {}
            }
            out.extend_from_slice(&f.join(&b","[..]));
            out.push(b'\n');
        }
        if next() % 3 == 0 { out.pop(); }
        out
    }

    #[test]
    fn fixed_sequence_matches_scalar_decoder() {
        let start = parse_ts(b"2028-02-20T00:00:00Z").unwrap();
        let w = Window { start, end: start + 14 * DAY_MS, click_start: start - WEEK_MS, start_day: start.div_euclid(DAY_MS) as i32 };
        let origin = w.start_day as i64 * DAY_MS - WEEK_MS;
        let clock = Clock::new(w.start, w.end);
        let prod_cat: Vec<u8> = (0..60_000u32).map(|i| if i % 97 == 0 { 255 } else { (i % 19) as u8 + 1 }).collect();
        let (prod_other, camp_by_num, camp_other) = (HashMap::new(), Vec::new(), HashMap::new());
        let cx = Ctx { prod_cat: &prod_cat, unknown: 0, prod_other: &prod_other, camp_by_num: &camp_by_num, camp_other: &camp_other, window: &w };
        let shared = Shared { slow: Mutex::new(SlowIntern::new()) };
        let b = Bits { user: 4, id: 5 };
        let dec = Decoder::new(&cx, &clock, &shared, origin, b);
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let fresh = || Scan { ev: bins(16), cl: Vec::new(), cand: bins(32), wide: Vec::new(), links: Vec::new() };
        let (mut total, mut slow) = (0usize, 0usize);
        for round in 0..60 {
            let rows = 1 + (next() % 2500) as usize;
            let buf = synth_events(&mut next, &w, rows);
            let (mut a, mut c) = (fresh(), fresh());
            batched::<7>(&buf, |rows| dec.events(rows, &mut a));
            let mut l = Lists::new();
            SLOW_ROWS.store(0, AtomicOrdering::Relaxed);
            dec.event_rows(&buf, &mut c, &mut l);
            total += rows;
            slow += SLOW_ROWS.load(AtomicOrdering::Relaxed);
            let ev = |s: &Scan| s.ev.iter().map(|v| v.iter().map(|e| (e.key, e.val)).collect::<Vec<_>>()).collect::<Vec<_>>();
            let cand = |s: &Scan| s.cand.iter().map(|v| v.iter().map(|c| (c.h, c.sec, c.loc)).collect::<Vec<_>>()).collect::<Vec<_>>();
            let wide = |s: &Scan| s.wide.iter().map(|v| v.iter().map(|x| (x.h, x.at, x.at_row)).collect::<Vec<_>>()).collect::<Vec<_>>();
            assert_eq!(ev(&a), ev(&c), "events, round {round}");
            assert_eq!(cand(&a), cand(&c), "candidates, round {round}");
            assert_eq!(wide(&a), wide(&c), "wide candidates, round {round}");
        }
        // Most synthetic rows have a generator shape: the fixed sequence must own them.
        assert!(slow * 2 < total, "exact path took {slow} of {total} rows");
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
        report("scan_events", &mut || {
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            batched::<7>(body, |rows| dec.events(rows, &mut out));
            std::hint::black_box(&out.ev);
        });
        report("stage1", &mut || {
            let mut l = Lists::new();
            let mut acc = 0usize;
            let mut start = 0;
            while start < body.len() {
                let end = (start + WINDOW).min(body.len());
                let (nc, nl) = wire::scan_window(body, start, end, &mut l);
                acc += nc;
                start = if nl == 0 { end } else { start + (l.lines[nl - 1] & !POISON) as usize + 1 };
            }
            std::hint::black_box(acc);
        });
        report("event_rows", &mut || {
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            let mut l = Lists::new();
            SLOW_ROWS.store(0, AtomicOrdering::Relaxed);
            dec.event_rows(body, &mut out, &mut l);
            std::hint::black_box(&out.ev);
            eprintln!("slow rows: {}", SLOW_ROWS.load(AtomicOrdering::Relaxed));
        });
        eprintln!("rows={rows_n} bytes={}", body.len());
    }
}
