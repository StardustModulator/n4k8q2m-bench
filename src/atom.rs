//! Bit 63 tags an interned string. A fast key, from high to low, contains a
//! clear tag bit, a 3-bit prefix rank, eight 7-bit ASCII bytes, and a 4-bit
//! total length. The bytes are in reading order, padded on the right with NUL.
//! Padding sorts before every nonzero byte; the length breaks ties such as
//! `u` vs `u\0`. Thus unsigned key order is exactly bytewise string order.

use std::cmp::Ordering;

const SLOW_TAG: u64 = 1 << 63;
const ASCII: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const PREFIXES: &[u8; 8] = b"\0EKUeku\0"; // Ranks 1..=6 in ASCII order.
const RANK: [u8; 256] = {
    let mut t = [0; 256];
    let mut i = 1;
    while i <= 6 {
        t[PREFIXES[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// Remove the (already checked) zero high bit of each byte. Shifts and masks
/// rather than BMI2 PEXT/PDEP: those are microcoded and very slow on AMD Zen 1
/// and 2, which also report x86-64-v3.
#[inline(always)]
fn gather_ascii(x: u64) -> u64 {
    // Combine adjacent groups of 7, then 14, then 28 bits.
    let x = (x & 0x00ff_00ff_00ff_00ff) | ((x & 0xff00_ff00_ff00_ff00) >> 1);
    let x = (x & 0x0000_ffff_0000_ffff) | ((x & 0xffff_0000_ffff_0000) >> 2);
    u64::from(x as u32) | ((x >> 32) << 28)
}

#[inline]
fn scatter_ascii(x: u64) -> u64 {
    let x = (x & 0x0fff_ffff) | ((x >> 28) << 32);
    let x = (x & 0x0000_3fff_0000_3fff) | ((x << 2) & 0x3fff_0000_3fff_0000);
    (x & 0x007f_007f_007f_007f) | ((x << 1) & 0x7f00_7f00_7f00_7f00)
}

/// Accept `E/K/U/e/k/u` followed by zero through eight arbitrary ASCII bytes.
#[inline(always)]
pub(super) fn try_pack(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if !(1..=9).contains(&n) {
        return None;
    }
    let rank = RANK[s[0] as usize];
    if rank == 0 {
        return None;
    }
    let word = if let Some(tail) = s.last_chunk::<8>() {
        // One in-bounds word load for both common sizes. For n=8, shift out
        // the prefix; for n=9, the word already contains exactly the suffix.
        let word = u64::from_be_bytes(*tail);
        if n == 8 {
            word << 8
        } else {
            word
        }
    } else {
        // Fixed-size loads keep even the short case free of a memcpy call.
        match s[1..] {
            [a, b, c, d, e, f] => u64::from_be_bytes([a, b, c, d, e, f, 0, 0]),
            [a, b, c, d, e] => u64::from_be_bytes([a, b, c, d, e, 0, 0, 0]),
            [a, b, c, d] => u64::from_be_bytes([a, b, c, d, 0, 0, 0, 0]),
            [a, b, c] => u64::from_be_bytes([a, b, c, 0, 0, 0, 0, 0]),
            [a, b] => u64::from_be_bytes([a, b, 0, 0, 0, 0, 0, 0]),
            [a] => u64::from(a) << 56,
            _ => 0, // Empty suffix.
        }
    };
    if word & !ASCII != 0 {
        return None;
    }
    Some((u64::from(rank) << 60) | (gather_ascii(word) << 4) | n as u64)
}

/// `try_pack` of the `n` bytes at `p` for n = 8 or 9, with its outcome as a
/// flag, for the fixed-sequence row decoder.
///
/// # Safety
/// `p` must point at `n` readable bytes, and `n` must be 8 or 9.
#[inline(always)]
pub(super) unsafe fn pack_raw(p: *const u8, n: u32) -> (u64, bool) {
    debug_assert!(n == 8 || n == 9);
    // SAFETY: the last eight of the n bytes.
    let (rank, tail) = unsafe { (RANK[*p as usize], p.add(n as usize - 8).cast::<[u8; 8]>().read_unaligned()) };
    // n = 8: the word holds the prefix too; shift it out (see `try_pack`).
    let word = u64::from_be_bytes(tail) << (8 * (9 - n));
    ((u64::from(rank) << 60) | (gather_ascii(word) << 4) | u64::from(n), (rank != 0) & (word & !ASCII == 0))
}

/// Write the original bytes of a fast key to `buf[..returned_length]`.
#[inline]
pub(super) fn write_id(key: u64, buf: &mut [u8; 24]) -> usize {
    debug_assert!(key != 0 && !is_slow(key));
    buf[0] = PREFIXES[((key >> 60) & 7) as usize];
    let suffix = (key >> 4) & ((1 << 56) - 1);
    buf[1..9].copy_from_slice(&scatter_ascii(suffix).to_be_bytes());
    key as usize & 15
}

#[inline]
pub(super) fn packed_cmp(a: u64, b: u64) -> Ordering {
    debug_assert!(a != 0 && b != 0 && !is_slow(a) && !is_slow(b));
    a.cmp(&b)
}

#[inline]
pub(super) fn is_slow(key: u64) -> bool {
    key & SLOW_TAG != 0
}

#[inline]
pub(super) fn slow_key(index: u32) -> u64 {
    SLOW_TAG | u64::from(index)
}

#[inline]
pub(super) fn slow_index(key: u64) -> usize {
    debug_assert!(is_slow(key));
    key as u32 as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &[u8]) -> u64 {
        let k = try_pack(s).unwrap();
        assert_ne!(k, 0);
        assert!(!is_slow(k));
        let mut buf = [0xee; 24];
        let n = write_id(k, &mut buf);
        assert_eq!(&buf[..n], s);
        k
    }

    #[test]
    fn exact_bytes_at_every_length_and_position() {
        for &prefix in b"EKUeku" {
            let mut s = *b"_0a9!Z-\0K";
            s[0] = prefix;
            for n in 1..=9 {
                roundtrip(&s[..n]);
                for i in 1..n {
                    let original = s[i];
                    for c in 0..=127 {
                        s[i] = c;
                        roundtrip(&s[..n]);
                    }
                    s[i] = original;
                }
            }
        }
    }

    #[test]
    fn order_matches_byte_slices() {
        let mut rows = Vec::new();
        let mut rand = 123456789u32;
        for &prefix in b"uKeUkE" {
            for n in 1..=9 {
                // Include prefix chains differing only in trailing NULs.
                for suffix in [*b"\0\0\0\0\0\0\0\0", *b"0\0\0\0\0\0\0\0", *b"00aAZ!-\x7f"] {
                    let mut s = [prefix; 9];
                    s[1..].copy_from_slice(&suffix);
                    rows.push(s[..n].to_vec());
                }
                for _ in 0..32 {
                    let mut s = [prefix; 9];
                    for c in &mut s[1..n] {
                        rand = rand.wrapping_mul(1664525).wrapping_add(1013904223);
                        *c = (rand >> 25) as u8;
                    }
                    rows.push(s[..n].to_vec());
                }
            }
        }
        rows.sort_unstable();
        for w in rows.windows(2) {
            let (a, b) = (roundtrip(&w[0]), roundtrip(&w[1]));
            let expected = w[0].cmp(&w[1]);
            assert_eq!(packed_cmp(a, b), expected);
            assert_eq!(packed_cmp(b, a), expected.reverse());
        }
    }

    #[test]
    fn fallback_and_interned_keys() {
        for s in [&b""[..], b"u123456789", b"  e1", "eé".as_bytes()] {
            assert_eq!(try_pack(s), None);
        }
        for prefix in 0..=255 {
            assert_eq!(try_pack(&[prefix]).is_some(), b"EKUeku".contains(&prefix));
        }
        for n in 2..=9 {
            for i in 1..n {
                let mut s = *b"k00000000";
                for c in [0x80, 0xff] {
                    s[i] = c;
                    assert_eq!(try_pack(&s[..n]), None);
                }
            }
        }
        for idx in [0, 1, 1 << 31, u32::MAX] {
            let k = slow_key(idx);
            assert_ne!(k, 0);
            assert!(is_slow(k));
            assert_eq!(slow_index(k), idx as usize);
        }
    }
}
