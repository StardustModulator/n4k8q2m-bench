//! Window-local acceleration for absolute timestamps. The window determines
//! which calendar dates to cache, never which instants the parser accepts.

use super::time::{civil_from_days, format_day, parse_ts, DAY_MS};

/// One cached date: its "YYYY-MM-" word, then "DDT" in the low three bytes
/// of `hd` and the day number (as an i32) above them. Sixteen bytes, so a
/// hit costs one aligned load.
#[derive(Clone, Copy)]
struct Day {
    lo: u64,
    hd: u64,
}

#[repr(align(16))]
pub(super) struct Clock {
    days: [Day; 64],
    #[cfg(test)]
    origin: i64, // first day of the window
}

/// The cache slot of a date whose "YYYY-MM-" word is `lo` and whose
/// "DDT?" bytes are `hi`.
#[inline(always)]
fn slot(lo: u64, hi: u32) -> usize {
    // The high byte of this product is (10 * b[8] + b[9]) mod 256.
    // Its low five bits distinguish all days of a month. Month parity keeps
    // adjacent months apart, including December and January. This is only an
    // index; neither a matching day number nor a hash hit establishes a date.
    let dd = (hi as u16).wrapping_mul(2561) >> 8;
    ((lo >> 48) as usize & 1) << 5 | usize::from(dd & 31)
}

impl Clock {
    pub(super) fn new(start: i64, end: i64) -> Self {
        // Even the default entry represents a real date: no arbitrary input
        // can match an uninitialized/all-zero tag and be mistaken for a hit.
        let entry = |text: &[u8; 13], d: i32| Day {
            lo: word(&text[..8]),
            hd: u64::from(u32::from_le_bytes(text[8..12].try_into().unwrap()) & 0xff_ffff) | u64::from(d as u32) << 24,
        };
        let mut days = [entry(b"1970-01-01T00", 0); 64];
        // Include the click lookback and the preceding local day for F2.
        // Leave room after the window for offsets and late ingest dates too.
        // Bound the work even if passed the entire i64 range.
        let first = start.div_euclid(DAY_MS) - 8;
        let last = (end.div_euclid(DAY_MS) + 8).min(first + days.len() as i64 - 1);
        for d in first..=last {
            let Ok(day) = i32::try_from(d) else { continue };
            if !(0..=9999).contains(&civil_from_days(d).year) {
                continue; // format_day writes a four-digit unsigned year
            }
            let mut text = *b"1970-01-01T00";
            format_day(day, (&mut text[..10]).try_into().unwrap());
            let e = entry(&text, day);
            days[slot(e.lo, e.hd as u32)] = e;
        }
        #[cfg(test)]
        return Self { days, origin: start.div_euclid(DAY_MS) };
        #[cfg(not(test))]
        Self { days }
    }

    /// F1, F2 or unsigned ASCII epoch milliseconds (F3).
    #[inline(always)]
    pub(super) fn parse(&self, b: &[u8]) -> Option<i64> {
        let p = b.as_ptr();
        // SAFETY: each arm reads only within the checked length.
        unsafe {
            match b.len() {
                20 if *p.add(19) == b'Z' => Some(self.iso::<20>(p)? * 1000),
                25 if matches!(*p.add(19), b'+' | b'-') => Some(self.iso::<25>(p)? * 1000),
                _ => epoch_ms(b),
            }
        }
    }

    /// `parse` for a `ts` field whose record extends at least 32 bytes past
    /// the field's start, without a data-dependent branch: the F1/F2 and F3
    /// readings are both computed from one set of loads and the length picks
    /// one. Lengths with no fast reading (0..=8, 17..=19, 21..=24, 26..) take
    /// the exact path. A cached date is the only way to a valid ISO reading
    /// here: the cache covers every day that can reach the window even
    /// through a zone offset, so an uncached date is out of the window and
    /// its exact instant could not be used anyway.
    ///
    /// # Safety
    /// 32 bytes must be readable at `p`.
    #[inline(always)]
    pub(super) unsafe fn parse_fast(&self, p: *const u8, n: usize) -> Option<i64> {
        let digits = n.wrapping_sub(9) <= 7;
        if !(digits || n == 20 || n == 25) {
            // SAFETY: n bytes are readable: n <= 32 here or the field lies in
            // the record, which the caller measured.
            return parse_cold(self, unsafe { std::slice::from_raw_parts(p, n) });
        }
        // SAFETY: every offset is below 32 (max(n, 8) - 8 <= 24).
        let (w0, w1, hms, tail, z) = unsafe {
            (load::<u64>(p), load::<u32>(p.add(8)), load::<u64>(p.add(11)), load::<u64>(p.add(17)), load::<u64>(p.add(n.max(8) - 8)))
        };
        // ISO: cached date, HH:MM:SS, then 'Z' or a checked "HH:MM" offset.
        let day = self.days[slot(w0, w1)];
        let day_ok = day.lo == w0 && day.hd as u32 & 0xff_ffff == w1 & 0xff_ffff;
        let (t, t_ok) = hms_pair(hms);
        let local = ((day.hd >> 24) as u32 as i32 as i64) * (DAY_MS / 1000) + t;
        let b19 = (tail >> 16) as u8;
        let (off, off_ok) = hms_pair((tail >> 24) | (HMS_MIN & (u64::MAX << 40)));
        let zone_ok = if n == 25 { (b19 == b'+' || b19 == b'-') && off_ok } else { b19 == b'Z' };
        let signed = if b19 == b'+' { -off } else { off };
        let iso = (local + if n == 25 { signed } else { 0 }) * 1000;
        let iso_ok = day_ok && t_ok && zone_ok;
        // F3: the first n - 8 digits left-padded to eight lanes, then the
        // last eight, with every lane checked for '0'..='9'.
        const BIAS: u64 = LOW7 - u64::from_le_bytes(*b"99999999");
        let hi = w0.wrapping_sub(ZEROS);
        let lo = z.wrapping_sub(ZEROS);
        let f3_ok = digits && (hi | lo | w0.wrapping_add(BIAS) | z.wrapping_add(BIAS)) & HIGH == 0;
        let prefix = hi << ((16 - n.clamp(9, 16)) * 8);
        let f3 = (decimal8(prefix) * 100_000_000 + decimal8(lo)) as i64;
        let (v, ok) = if digits { (f3, f3_ok) } else { (iso, iso_ok) };
        ok.then_some(v)
    }

    /// The window start the cache was built for, as a day boundary (tests).
    #[cfg(test)]
    pub(super) fn days_origin(&self) -> i64 {
        self.origin * DAY_MS
    }

    /// The ingest column has its own F1-only entry point. In particular an
    /// old or very late ingest still participates in dedup at its real time.
    #[cfg(test)]
    pub(super) fn ingest(&self, b: &[u8]) -> Option<i64> {
        Some(self.ingest_s(b)? * 1000)
    }

    /// `ingest` in whole seconds, which is exact: F1 has no fraction.
    #[inline(always)]
    pub(super) fn ingest_s(&self, b: &[u8]) -> Option<i64> {
        if b.len() != 20 || b[19] != b'Z' {
            return None;
        }
        // SAFETY: 20 bytes.
        unsafe { self.iso::<20>(b.as_ptr()) }
    }

    /// Seconds since the epoch of the `N`-byte ISO form at `p`.
    ///
    /// # Safety
    /// `N` bytes must be readable at `p`.
    #[inline(always)]
    unsafe fn iso<const N: usize>(&self, p: *const u8) -> Option<i64> {
        // SAFETY: all offsets lie below 20 <= N.
        let (lo, hi, hms) = unsafe { (load(p), load::<u32>(p.add(8)), load::<u64>(p.add(11))) };
        let day = self.days[slot(lo, hi)];
        if day.lo != lo || day.hd as u32 & 0xff_ffff != hi & 0xff_ffff {
            // SAFETY: forwarded.
            return uncached_iso(unsafe { std::slice::from_raw_parts(p, N) });
        }
        let local = ((day.hd >> 24) as u32 as i32 as i64) * (DAY_MS / 1000) + hms_s(hms)?;
        if N == 25 {
            // Load exactly through the end of F2: "SS±HH:MM". Discard the
            // first three bytes and supply ":00" seconds for the same checked
            // HH:MM:SS decoder. Keep the offset in signed seconds.
            // SAFETY: bytes 17..25 of 25.
            let hm = unsafe { load::<u64>(p.add(17)) } >> 24;
            let offset = hms_s(hm | (HMS_MIN & (u64::MAX << 40)))?;
            // SAFETY: byte 19 of 25.
            Some(if unsafe { *p.add(19) } == b'+' { local - offset } else { local + offset })
        } else {
            Some(local)
        }
    }
}

/// An unaligned little-endian load.
///
/// # Safety
/// `size_of::<T>()` bytes must be readable at `p`.
#[inline(always)]
unsafe fn load<T: Copy>(p: *const u8) -> T {
    // SAFETY: forwarded.
    unsafe { p.cast::<T>().read_unaligned() }
}

// Keep the general calendar conversion off the row-parsing hot path. ISO
// forms are whole seconds (offsets are whole minutes), so the division is exact.
#[cold]
#[inline(never)]
fn uncached_iso(b: &[u8]) -> Option<i64> {
    parse_ts(b).map(|ms| ms / 1000)
}

#[inline(always)]
fn word(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}

const HIGH: u64 = 0x8080_8080_8080_8080;
const LOW7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const HMS_MIN: u64 = u64::from_le_bytes(*b"00:00:00");
const HMS_BIAS: u64 = LOW7 - u64::from_le_bytes(*b"29:59:59");
const ZEROS: u64 = u64::from_le_bytes(*b"00000000");

#[inline(always)]
fn hms_s(raw: u64) -> Option<i64> {
    let (v, ok) = hms_pair(raw);
    ok.then_some(v)
}

/// Seconds of "HH:MM:SS" and whether it was one, without a branch.
#[inline(always)]
fn hms_pair(raw: u64) -> (i64, bool) {
    // In each lane require MIN <= byte <= MAX (the two colons have min=max).
    // Valid lanes cannot carry/borrow into a neighbor, so the *first* invalid
    // lane necessarily contributes a high bit to one of these two words.
    let digits = raw.wrapping_sub(HMS_MIN);
    let bad = (digits | raw.wrapping_add(HMS_BIAS)) & HIGH;
    // 2561 = 10*256+1: each ones byte receives ten times its tens byte.
    let pairs = digits.wrapping_mul(2561) >> 8;
    let h = i64::from(pairs as u8);
    let m = i64::from((pairs >> 24) as u8);
    let s = i64::from((pairs >> 48) as u8);
    ((h * 60 + m) * 60 + s, bad == 0 && h <= 23)
}

/// Any form, any length, off the hot path.
#[cold]
#[inline(never)]
fn parse_cold(c: &Clock, b: &[u8]) -> Option<i64> {
    c.parse(b)
}

#[inline(always)]
fn epoch_ms(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if !(9..=16).contains(&n) {
        return other_epoch_ms(b);
    }
    // Two in-bounds loads cover all bytes, even when they overlap. Left-pad
    // the first n-8 digits to eight lanes; the last eight need no adjustment.
    const BIAS: u64 = LOW7 - u64::from_le_bytes(*b"99999999");
    let a = word(&b[..8]);
    let z = word(&b[n - 8..]);
    let hi = a.wrapping_sub(ZEROS);
    let lo = z.wrapping_sub(ZEROS);
    if (hi | lo | a.wrapping_add(BIAS) | z.wrapping_add(BIAS)) & HIGH != 0 {
        return None;
    }
    let prefix = hi << ((16 - n) * 8);
    Some((decimal8(prefix) * 100_000_000 + decimal8(lo)) as i64)
}

/// Reduce eight little-endian digit lanes (values 0..9) to one integer.
#[inline(always)]
pub(super) fn decimal8(digits: u64) -> u64 {
    let pairs = digits.wrapping_mul(10).wrapping_add(digits >> 8);
    // Select pairs 0 and 2, then 1 and 3. In the high 32 bits the products
    // are weighted by 10^6, 10^2 and by 10^4, 1 respectively.
    const MASK: u64 = 0x0000_00ff_0000_00ff;
    let even = (pairs & MASK).wrapping_mul((1_000_000u64 << 32) | 100);
    let odd = ((pairs >> 16) & MASK).wrapping_mul((10_000u64 << 32) | 1);
    even.wrapping_add(odd) >> 32
}

#[cold]
#[inline(never)]
fn other_epoch_ms(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    // Any digits in excess of 19 must be leading ASCII zeroes. This also
    // handles an arbitrarily long representation of zero without allocation.
    let (padding, digits) = b.split_at(b.len().saturating_sub(19));
    if padding.iter().any(|&c| c != b'0') {
        return None;
    }
    let mut n = 0u64; // every 19-digit number fits u64 (but not always i64)
    for &c in digits {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        n = n * 10 + u64::from(d);
    }
    i64::try_from(n).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(s: &str) -> Clock {
        let start = parse_ts(s.as_bytes()).unwrap();
        Clock::new(start, start + 14 * DAY_MS)
    }

    #[test]
    fn calendar_transitions_and_cache_misses() {
        for anchor in [
            "0000-01-01T00:00:00Z",
            "1900-02-24T12:34:56Z", // century, non-leap
            "1969-12-29T23:59:59Z",
            "2000-02-24T00:00:00Z", // century, leap
            "2407-12-26T00:00:00Z",
            "9999-12-26T00:00:00Z",
        ] {
            let c = window(anchor);
            let d0 = parse_ts(anchor.as_bytes()).unwrap().div_euclid(DAY_MS);
            // Visit both sides of the cache, including different years which
            // reuse a day/month slot. Check against the actual epoch day, too.
            for delta in (-12..=25).chain([-731, 731]) {
                let d = d0 + delta;
                if !(0..=9999).contains(&civil_from_days(d).year) {
                    continue;
                }
                let mut tag = [0; 10];
                format_day(d as i32, &mut tag);
                let date = std::str::from_utf8(&tag).unwrap();
                for (zone, offset_min) in
                    [("Z", 0), ("-00:00", 0), ("+23:59", 1439), ("-23:59", -1439)]
                {
                    let s = format!("{date}T00:01:02{zone}");
                    let want = Some(d * DAY_MS + 62_000 - offset_min * 60_000);
                    assert_eq!(c.parse(s.as_bytes()), want, "{s}");
                    assert_eq!(c.parse(s.as_bytes()), parse_ts(s.as_bytes()), "{s}");
                    if zone == "Z" {
                        assert_eq!(c.ingest(s.as_bytes()), want, "{s}");
                    } else {
                        assert_eq!(c.ingest(s.as_bytes()), None);
                    }
                }
            }
        }
    }

    #[test]
    fn every_iso_byte_is_checked() {
        let c = window("2000-02-25T00:00:00Z");
        for original in [
            &b"2000-02-29T23:59:59Z"[..],
            &b"2000-02-29T23:59:59-23:59"[..],
        ] {
            let mut s = original.to_vec();
            for i in 0..s.len() {
                for byte in 0..=u8::MAX {
                    s[i] = byte;
                    let want = parse_ts(&s);
                    assert_eq!(c.parse(&s), want, "byte {byte} at {i} in {original:?}");
                    if original.len() == 20 {
                        assert_eq!(c.ingest(&s), want);
                    }
                }
                s[i] = original[i];
            }
        }
        for bad in [
            "1900-02-29T00:00:00Z",
            "2000-02-30T00:00:00Z",
            "2000-04-31T00:00:00+00:00",
            "2000-02-29T12:34:56.000Z",
            "2000-02-29T12:34:56.123+01:00",
            "2000-02-29T12:34:56+0100",
        ] {
            assert_eq!(c.parse(bad.as_bytes()), None, "{bad}");
        }
    }

    /// `parse_fast` on a field inside a longer record agrees with `parse`.
    fn fast(c: &Clock, b: &[u8]) -> Option<i64> {
        let mut rec = b.to_vec();
        rec.extend_from_slice(b",2026-01-01T00:00:00Z,view,p1,100\n");
        unsafe { c.parse_fast(rec.as_ptr(), b.len()) }
    }

    #[test]
    fn fast_path_matches_parse() {
        let c = window("2000-02-25T00:00:00Z");
        for original in [
            &b"2000-02-29T23:59:59Z"[..], b"2000-02-29T23:59:59-23:59", b"2000-02-29T23:59:59+00:00",
            b"951868799000", b"0951868799000", b"000000951868799000", b"95186879", b"00000000000000000000951868799000",
        ] {
            let mut s = original.to_vec();
            for i in 0..s.len() {
                for byte in 0..=u8::MAX {
                    s[i] = byte;
                    let want = c.parse(&s);
                    let got = fast(&c, &s);
                    // Outside the cached dates the fast path reports a miss;
                    // such an instant is out of every window the cache serves.
                    let cached = want.is_some_and(|ms| (-9 * DAY_MS..=23 * DAY_MS).contains(&(ms - c.days_origin())));
                    if cached || want.is_none() || s.len() != original.len() || !(s.len() == 20 || s.len() == 25) {
                        assert_eq!(got, want, "byte {byte} at {i} in {original:?}");
                    }
                }
                s[i] = original[i];
            }
            for n in 0..original.len() {
                let t = &original[..n];
                let want = c.parse(t);
                if !(n == 20 || n == 25) { assert_eq!(fast(&c, t), want, "{t:?}"); }
            }
        }
    }

    fn reference_epoch(b: &[u8]) -> Option<i64> {
        if !b.iter().all(u8::is_ascii_digit) {
            return None;
        }
        std::str::from_utf8(b).ok()?.parse().ok()
    }

    #[test]
    fn epoch_millis_digits_and_padding() {
        let c = Clock::new(-1, 1);
        for s in [
            "",
            "0",
            "001",
            "86400000",
            "1791071890001",
            "9223372036854775807",
            "9223372036854775808",
            "9999999999999999999",
            "00000000000000000000000000010000000000000000000",
            "+1",
            "-5",
            "1234567890123.0",
            "1.79e12",
            "1234567890123\r",
            "٢٥٠",
            "１２３",
        ] {
            assert_eq!(c.parse(s.as_bytes()), reference_epoch(s.as_bytes()), "{s}");
        }
        for n in [0, 1, 9_012_345_678_901, i64::MAX] {
            for width in [19, 20, 25, 64, 4096] {
                let s = format!("{n:0width$}");
                assert_eq!(c.parse(s.as_bytes()), Some(n));
            }
        }
        // Exercise all overlapping-load widths and every position, including
        // non-digits that share the low nibble of an ASCII digit.
        for width in 1..=19 {
            let original = &b"9078612345098123476"[..width];
            let mut b = original.to_vec();
            for i in 0..width {
                for v in [0, b'/', b'0', b'5', b'9', b':', b'P', 0x80, 0xff] {
                    b[i] = v;
                    assert_eq!(c.parse(&b), reference_epoch(&b), "{b:?}");
                }
                b[i] = original[i];
            }
        }
    }

    #[test]
    fn extreme_window_bounds_do_not_limit_ingest() {
        for (start, end) in [
            (i64::MIN, i64::MIN + 1),
            (i64::MIN, i64::MAX),
            (i64::MAX - 1, i64::MAX),
        ] {
            let c = Clock::new(start, end);
            for b in [b"0000-02-29T12:34:56Z", b"9999-12-31T23:59:59Z"] {
                assert_eq!(c.ingest(b), parse_ts(b));
            }
            assert_eq!(c.ingest(b"00000000000000000001"), None);
        }
    }
}
