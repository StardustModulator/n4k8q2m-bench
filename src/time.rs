//! Timestamp forms F1, F2 and F3. Calendar checks reject impossible dates.

pub const SECOND: i64 = 1_000;
pub const DAY_MS: i64 = 86_400_000;
pub const WEEK_MS: i64 = 7 * DAY_MS;
pub const GAP_MS: i64 = 1_800_000;
pub const LATE_MS: i64 = DAY_MS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    pub year: i64,
    pub month: i64,
    pub day: i64,
}

#[inline]
fn leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

#[inline]
fn days_in_month(year: i64, month: i64) -> i64 {
    const DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if month == 2 && leap(year) {
        29
    } else {
        DAYS[(month as usize) - 1]
    }
}

const fn civil_days(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    era * 146_097 + year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year - 719_468
}

// Jan 1 for 2005..=2044. The generator's windows sit inside this span.
const JAN1: [i64; 40] = {
    let mut out = [0i64; 40];
    let mut i = 0;
    while i < 40 {
        out[i] = civil_days(2005 + i as i64, 1, 1);
        i += 1;
    }
    out
};

const CUM_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

/// Howard Hinnant `days_from_civil`. `None` when the calendar date is impossible.
#[inline]
pub fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let dim = days_in_month(year, month);
    if day < 1 || day > dim {
        return None;
    }
    Some(civil_days(year, month, day))
}

#[inline(always)]
fn fast_day(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let leap = leap(year);
    let dim = if month == 2 {
        28 + i64::from(leap)
    } else if matches!(month, 4 | 6 | 9 | 11) {
        30
    } else {
        31
    };
    if day < 1 || day > dim {
        return None;
    }
    if (2005..2045).contains(&year) {
        let mut off = CUM_DAYS[(month as usize) - 1] + day - 1;
        if month > 2 && leap {
            off += 1;
        }
        Some(JAN1[(year - 2005) as usize] + off)
    } else {
        Some(civil_days(year, month, day))
    }
}

pub fn civil_from_days(days: i64) -> Civil {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    Civil {
        year: year_of_era + era * 400 + i64::from(month <= 2),
        month,
        day: day_of_year - (153 * month_index + 2) / 5 + 1,
    }
}

#[inline(always)]
fn two_digits(a: u8, b: u8) -> Option<i64> {
    let a = a.wrapping_sub(b'0');
    let b = b.wrapping_sub(b'0');
    if a > 9 || b > 9 {
        return None;
    }
    Some(a as i64 * 10 + b as i64)
}

/// High nibbles of eight ASCII digits collapse to `0x33`.
#[inline(always)]
fn eight_digits(word: u64) -> bool {
    let mixed = (word & 0xF0F0_F0F0_F0F0_F0F0)
        | (((word.wrapping_add(0x0606_0606_0606_0606)) & 0xF0F0_F0F0_F0F0_F0F0) >> 4);
    mixed == 0x3333_3333_3333_3333
}

#[inline(always)]
fn pair_at(word: u64, byte: u32) -> i64 {
    let v = word >> (byte * 8);
    ((v & 0x0F) * 10 + ((v >> 8) & 0x0F)) as i64
}

/// `YYYY-MM-DDTHH:MM:SS` at the front of `s` (19 bytes). No trailing `Z` check.
#[inline(always)]
fn clock_ms(s: &[u8]) -> Option<i64> {
    if s.len() < 19 {
        return None;
    }
    let w0 = u64::from_le_bytes(s[0..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(s[8..16].try_into().unwrap());
    // Bytes 4 and 7 are `-`. Byte 10 (`w1` byte 2) is `T`. Byte 13 (`w1` byte 5) and 16 are `:`.
    const M0: u64 = (0xFFu64 << 32) | (0xFFu64 << 56);
    const V0: u64 = (0x2Du64 << 32) | (0x2Du64 << 56);
    const M1: u64 = (0xFFu64 << 16) | (0xFFu64 << 40);
    const V1: u64 = (0x54u64 << 16) | (0x3Au64 << 40);
    const Z0: u64 = (0x30u64 << 32) | (0x30u64 << 56);
    const Z1: u64 = (0x30u64 << 16) | (0x30u64 << 40);
    if (w0 & M0) != V0 || (w1 & M1) != V1 || s[16] != b':' {
        return None;
    }
    let d0 = (w0 & !M0) | Z0;
    let d1 = (w1 & !M1) | Z1;
    if !eight_digits(d0) || !eight_digits(d1) {
        return None;
    }
    let sec_hi = s[17].wrapping_sub(b'0');
    let sec_lo = s[18].wrapping_sub(b'0');
    if sec_hi > 9 || sec_lo > 9 {
        return None;
    }
    let year = pair_at(d0, 0) * 100 + pair_at(d0, 2);
    let month = pair_at(d0, 5);
    let day = pair_at(d1, 0);
    let hour = pair_at(d1, 3);
    let minute = pair_at(d1, 6);
    let second = sec_hi as i64 * 10 + sec_lo as i64;
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let days = fast_day(year, month, day)?;
    Some(days * DAY_MS + ((hour * 60 + minute) * 60 + second) * SECOND)
}

#[inline(always)]
fn f3_ms(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 || n > 19 {
        return None;
    }
    let mut v: u64 = 0;
    let mut i = 0;
    // 8-digit SWAR chunks (Lemire).
    while i + 8 <= n {
        let word = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        if !eight_digits(word) {
            return None;
        }
        let mut chunk = (word & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561) >> 8;
        chunk = (chunk & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601) >> 16;
        chunk = (chunk & 0x0000_FFFF_0000_FFFF).wrapping_mul(4_294_967_296_0001) >> 32;
        v = v.checked_mul(100_000_000)?.checked_add(chunk)?;
        i += 8;
    }
    while i < n {
        let d = s[i].wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as u64)?;
        i += 1;
    }
    i64::try_from(v).ok()
}

/// F1 `YYYY-MM-DDTHH:MM:SSZ`, F2 with `±HH:MM`, or F3 digits (leading zeros allowed).
#[inline(always)]
pub fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 20 && s[19] == b'Z' {
        return clock_ms(s);
    }
    if n == 25 && (s[19] == b'+' || s[19] == b'-') && s[22] == b':' {
        let local = clock_ms(s)?;
        let hh = two_digits(s[20], s[21])?;
        let mm = two_digits(s[23], s[24])?;
        if hh > 23 || mm > 59 {
            return None;
        }
        let off = (hh * 60 + mm) * 60 * SECOND;
        let signed = if s[19] == b'-' { -off } else { off };
        return Some(local - signed);
    }
    if n == 0 || !s[0].is_ascii_digit() {
        return None;
    }
    f3_ms(s)
}

pub fn format_day(days: i32, out: &mut [u8; 10]) {
    let c = civil_from_days(days as i64);
    let y = c.year as u32;
    let m = c.month as u32;
    let d = c.day as u32;
    out[0] = b'0' + (y / 1000) as u8;
    out[1] = b'0' + ((y / 100) % 10) as u8;
    out[2] = b'0' + ((y / 10) % 10) as u8;
    out[3] = b'0' + (y % 10) as u8;
    out[4] = b'-';
    out[5] = b'0' + (m / 10) as u8;
    out[6] = b'0' + (m % 10) as u8;
    out[7] = b'-';
    out[8] = b'0' + (d / 10) as u8;
    out[9] = b'0' + (d % 10) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_known_instants() {
        assert_eq!(days_from_civil(1970, 1, 1), Some(0));
        assert_eq!(civil_from_days(0), Civil { year: 1970, month: 1, day: 1 });
        let f1 = parse_ts(b"2026-10-03T23:58:10Z").unwrap();
        let f2 = parse_ts(b"2026-10-03T19:58:10-04:00").unwrap();
        assert_eq!(f1, f2);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10-00:00").unwrap(), f1);
        assert_eq!(parse_ts(b"0001791071890000"), parse_ts(b"1791071890000"));
        for raw in [
            "1828714230000",
            "1828462575000",
            "1828627771436",
            "1829088235000",
        ] {
            assert_eq!(parse_ts(raw.as_bytes()).unwrap(), raw.parse::<i64>().unwrap(), "{raw}");
        }
        let z = parse_ts(b"2027-12-09T00:02:51Z").unwrap();
        assert_eq!(parse_ts(b"2027-12-09T00:02:51+00:00").unwrap(), z);
        assert_eq!(parse_ts(b"2027-12-09T04:21:07-04:00").unwrap(), parse_ts(b"2027-12-09T08:21:07Z").unwrap());
        assert_eq!(parse_ts(b"2027-12-11T07:06:43-09:30").unwrap(), parse_ts(b"2027-12-11T16:36:43Z").unwrap());
        assert_eq!(parse_ts(b"2027-12-23T02:49:37+08:45").unwrap(), parse_ts(b"2027-12-22T18:04:37Z").unwrap());
        assert!(parse_ts(b"2027-12-20T20:05:48Z").is_some());
        assert!(parse_ts(b"2027-12-09T16:15:48Z").is_some());
        assert!(parse_ts(b"2027-12-15T16:26:59Z").is_some());
        for day in 1..=31 {
            for hour in 0..24 {
                let s = format!("2027-12-{day:02}T{hour:02}:05:48Z");
                assert!(parse_ts(s.as_bytes()).is_some(), "{s}");
            }
        }
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert!(parse_ts(b"2027-02-29T00:00:00Z").is_none());
        assert!(parse_ts(b"2026-02-30T00:00:00Z").is_none());
        assert!(parse_ts(b"2026-13-01T00:00:00Z").is_none());
        assert!(parse_ts(b"2026-10-03T24:00:00Z").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:60Z").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:10z").is_none());
        assert!(parse_ts(b"2026-10-03t23:58:10Z").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:10+24:00").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:10+05:60").is_none());
        assert!(parse_ts(b"2026-10-03T23:58:10.0Z").is_none());
        assert!(parse_ts(b"-5").is_none());
        assert!(parse_ts(b"").is_none());
        let days = days_from_civil(2027, 12, 9).unwrap();
        let mut buf = [0u8; 10];
        format_day(days as i32, &mut buf);
        assert_eq!(&buf, b"2027-12-09");
    }
}
