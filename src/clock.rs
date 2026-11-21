//! Window-local acceleration for absolute timestamps. The window determines
//! which calendar dates to cache, never which instants the parser accepts.

use super::time::{civil_from_days, format_day, parse_ts, DAY_MS};

#[derive(Clone, Copy)]
struct Day {
    lo: u64,       // "YYYY-MM-"
    hi: u32,       // "DDT" in the low three bytes
    midnight: i64, // seconds
}

pub(super) struct Clock {
    days: [Day; 64],
}

#[inline(always)]
fn slot(b: &[u8]) -> usize {
    // The high byte of this product is (10 * b[8] + b[9]) mod 256.
    // Its low five bits distinguish all days of a month. Month parity keeps
    // adjacent months apart, including December and January. This is only an
    // index; neither a matching day number nor a hash hit establishes a date.
    let dd = u16::from_le_bytes([b[8], b[9]]).wrapping_mul(2561) >> 8;
    (usize::from(b[6] & 1) << 5) | usize::from(dd & 31)
}

/// The whole "YYYY-MM-DDT" prefix as two words; the hour digit is masked off.
#[inline(always)]
fn tag(b: &[u8]) -> (u64, u32) {
    (word(&b[..8]), u32::from_le_bytes(b[8..12].try_into().unwrap()) & 0xff_ffff)
}

impl Clock {
    pub(super) fn new(start: i64, end: i64) -> Self {
        // Even the default entry represents a real date: no arbitrary input
        // can match an uninitialized/all-zero tag and be mistaken for a hit.
        let (lo, hi) = tag(b"1970-01-01T00");
        let mut days = [Day { lo, hi, midnight: 0 }; 64];
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
            let (lo, hi) = tag(&text);
            days[slot(&text)] = Day { lo, hi, midnight: d * (DAY_MS / 1000) };
        }
        Self { days }
    }

    /// F1, F2 or unsigned ASCII epoch milliseconds (F3).
    #[inline(always)]
    pub(super) fn parse(&self, b: &[u8]) -> Option<i64> {
        match b.len() {
            20 if b[19] == b'Z' => Some(self.iso::<20>(b.try_into().unwrap())? * 1000),
            25 if matches!(b[19], b'+' | b'-') => Some(self.iso::<25>(b.try_into().unwrap())? * 1000),
            _ => epoch_ms(b),
        }
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
        let b: &[u8; 20] = b.try_into().ok()?;
        if b[19] != b'Z' {
            return None;
        }
        self.iso(b)
    }

    /// F1, F2 or F3 by one fixed instruction sequence: the calendar decode,
    /// the zone offset and the digit decode are all computed, and the
    /// field's length selects among them, so the form mix costs no branch.
    /// Returns the instant in milliseconds and whether it is certain. An
    /// uncertain result says only that `parse` must decide: a date outside
    /// the cached days, an F3 width outside 9..=16, or a bad timestamp.
    ///
    /// # Safety
    /// 25 bytes must be readable from `b.as_ptr()`, whatever `b.len()` is.
    #[inline(always)]
    pub(super) unsafe fn select(&self, b: &[u8]) -> (i64, bool) {
        let p = b.as_ptr();
        let n = b.len();
        // SAFETY: every offset read here is at most 16 (F3) or 17 (zone),
        // and reads eight bytes: inside the 25 the caller guarantees.
        let rd = |o: usize| unsafe { u64::from_le_bytes(p.add(o).cast::<[u8; 8]>().read_unaligned()) };
        let w0 = rd(0);
        let w8 = rd(8);
        // `slot` and `tag` over the words: bytes 6, 8, 9 and 0..12.
        let dd = (w8 as u16).wrapping_mul(2561) >> 8;
        let slot = (usize::from((w0 >> 48) as u8 & 1) << 5) | usize::from(dd & 31);
        let day = &self.days[slot];
        let dated = (day.lo == w0) & (day.hi == w8 as u32 & 0xff_ffff);
        let (hms, hms_ok) = hms_flag(rd(11));
        let local = day.midnight + hms;
        let zone = rd(17);
        let (off, off_ok) = hms_flag(zone >> 24 | (HMS_MIN & (u64::MAX << 40)));
        let sign = (zone >> 16) as u8; // byte 19
        let minus = sign == b'-';
        let iso25 = if minus { local + off } else { local - off };
        let ok20 = (n == 20) & dated & hms_ok & (sign == b'Z');
        let ok25 = (n == 25) & dated & hms_ok & off_ok & (minus | (sign == b'+'));
        // F3 as in `epoch_ms`: the last eight digits, and the rest left-padded.
        const ZERO: u64 = u64::from_le_bytes(*b"00000000");
        const BIAS: u64 = LOW7 - u64::from_le_bytes(*b"99999999");
        let z = rd(n.clamp(8, 16) - 8);
        let hi = w0.wrapping_sub(ZERO);
        let lo = z.wrapping_sub(ZERO);
        let digits = (hi | lo | w0.wrapping_add(BIAS) | z.wrapping_add(BIAS)) & HIGH == 0;
        let ok3 = digits & (9..=16).contains(&n);
        let prefix = hi.wrapping_shl(16u32.wrapping_sub(n as u32).wrapping_mul(8));
        let f3 = (decimal8(prefix) * 100_000_000 + decimal8(lo)) as i64;
        let ms = if n == 20 { local * 1000 } else if n == 25 { iso25 * 1000 } else { f3 };
        (ms, ok20 | ok25 | ok3)
    }

    /// Seconds since the epoch.
    #[inline(always)]
    fn iso<const N: usize>(&self, b: &[u8; N]) -> Option<i64> {
        let day = &self.days[slot(b)];
        if (day.lo, day.hi) != tag(b) {
            return uncached_iso(b);
        }
        let local = day.midnight + hms_s(word(&b[11..19]))?;
        if N == 25 {
            // Load exactly through the end of F2: "SS±HH:MM". Discard the
            // first three bytes and supply ":00" seconds for the same checked
            // HH:MM:SS decoder. Keep the offset in signed seconds.
            let hm = word(&b[17..25]) >> 24;
            let offset = hms_s(hm | (HMS_MIN & (u64::MAX << 40)))?;
            Some(if b[19] == b'+' {
                local - offset
            } else {
                local + offset
            })
        } else {
            Some(local)
        }
    }
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

#[inline(always)]
fn hms_s(raw: u64) -> Option<i64> {
    let (s, ok) = hms_flag(raw);
    ok.then_some(s)
}

/// "HH:MM:SS" as seconds, and whether every lane was in range.
#[inline(always)]
fn hms_flag(raw: u64) -> (i64, bool) {
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
    ((h * 60 + m) * 60 + s, (bad == 0) & (h <= 23))
}

#[inline(always)]
fn epoch_ms(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if !(9..=16).contains(&n) {
        return other_epoch_ms(b);
    }
    // Two in-bounds loads cover all bytes, even when they overlap. Left-pad
    // the first n-8 digits to eight lanes; the last eight need no adjustment.
    const ZERO: u64 = u64::from_le_bytes(*b"00000000");
    const BIAS: u64 = LOW7 - u64::from_le_bytes(*b"99999999");
    let a = word(&b[..8]);
    let z = word(&b[n - 8..]);
    let hi = a.wrapping_sub(ZERO);
    let lo = z.wrapping_sub(ZERO);
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

    /// `select` over a padded copy: certain results equal `parse`, and every
    /// in-window instant of a well-formed field is certain.
    fn select_padded(c: &Clock, s: &[u8]) -> (i64, bool) {
        let mut pad = vec![b','; s.len() + 32];
        pad[..s.len()].copy_from_slice(s);
        unsafe { c.select(&pad[..s.len()]) }
    }

    #[test]
    fn select_agrees_with_parse() {
        let c = window("2027-12-09T00:00:00Z");
        let mut cases: Vec<Vec<u8>> = Vec::new();
        for original in [
            &b"2027-12-20T23:59:59Z"[..],
            &b"2027-12-20T19:59:59-04:00"[..],
            &b"2027-12-20T19:59:59+13:30"[..],
            &b"1829088235000"[..],
            &b"0001829088235000"[..],
            &b"182908823"[..],
        ] {
            cases.push(original.to_vec());
            for i in 0..original.len() {
                for byte in [0, b' ', b'+', b'-', b'/', b'0', b'5', b'9', b':', b'T', b'Z', b'z', 0x80, 0xff] {
                    let mut s = original.to_vec();
                    s[i] = byte;
                    cases.push(s);
                }
            }
        }
        for s in ["", "1", "12345678", "12345678901234567", "00000000000000000000", "0000000000000000000000000",
                  "2027-12-20T23:59:59z", "2027-12-20t23:59:59Z", "2027-12-20T23:59:59.000Z", "2027-12-20T23:59:59+0400",
                  "2026-02-29T00:00:00Z", "2028-02-29T00:00:00Z", "2027-12-20T23:59:59-00:00", "1700-12-20T23:59:59Z"] {
            cases.push(s.as_bytes().to_vec());
        }
        for s in &cases {
            let want = parse_ts(s);
            let (got, sure) = select_padded(&c, s);
            let n = s.len();
            // The cache holds 2027-12-01 through 2027-12-31 for this window.
            let cached = n >= 10 && (&b"2027-12-01"[..]..=&b"2027-12-31"[..]).contains(&&s[..10]);
            let certain = want.is_some() && ((n == 20 || n == 25) && cached || (9..=16).contains(&n));
            assert_eq!(sure, certain, "{:?}", std::str::from_utf8(s));
            if sure {
                assert_eq!(Some(got), want, "{:?}", std::str::from_utf8(s));
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
