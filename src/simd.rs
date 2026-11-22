//! Next CSV structural byte: `,`, `\n`, or `"`.
//!
//! x86_64 built with AVX2 (`target-cpu=x86-64-v3` or `target_feature=+avx2`)
//! scans 32 unaligned bytes per step. aarch64 and every other target use a
//! scalar byte loop: on Apple Silicon a short field matches before a NEON
//! movemask pays for itself.

/// Append every index of `,`, `\n`, or `"` in `buf`.
///
/// x86-64-v3 uses AVX2 movemask. Other targets use an 8-byte SWAR scan.
/// Both are little-endian: the lowest set bit is the earliest byte.
pub fn push_structurals(buf: &[u8], out: &mut Vec<u32>) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        push_structurals_avx2(buf, out);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    push_structurals_swar(buf, out);
}

#[inline(always)]
fn swar_structural(word: u64) -> u64 {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let comma = word ^ (0x2Cu64 * ONES);
    let nl = word ^ (0x0Au64 * ONES);
    let quote = word ^ (0x22u64 * ONES);
    let hit = |x: u64| x.wrapping_sub(ONES) & !x & HIGH;
    hit(comma) | hit(nl) | hit(quote)
}

fn push_structurals_swar(buf: &[u8], out: &mut Vec<u32>) {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let mut i = 0usize;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len`, so the unaligned load stays inside `buf`.
        let word = unsafe { ptr.add(i).cast::<u64>().read_unaligned() };
        let mut mask = swar_structural(word);
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let at = i + (bit >> 3);
            // The subtract trick also lights up a neighbor of a real comma, such as `-`.
            let c = unsafe { *ptr.add(at) };
            if c == b',' || c == b'\n' || c == b'"' {
                out.push(at as u32);
            }
            mask &= mask - 1;
        }
        i += 8;
    }
    while i < len {
        let c = buf[i];
        if c == b',' || c == b'\n' || c == b'"' {
            out.push(i as u32);
        }
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn push_structurals_avx2(buf: &[u8], out: &mut Vec<u32>) {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_set1_epi8,
    };
    let len = buf.len();
    let ptr = buf.as_ptr();
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);
    let mut i = 0usize;
    while i + 32 <= len {
        // SAFETY: `i + 32 <= len`.
        let v = unsafe { _mm256_loadu_si256(ptr.add(i).cast()) };
        let eq = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(v, comma), _mm256_cmpeq_epi8(v, nl)),
            _mm256_cmpeq_epi8(v, quote),
        );
        let mut mask = _mm256_movemask_epi8(eq) as u32;
        while mask != 0 {
            let bit = mask.trailing_zeros();
            out.push(i as u32 + bit);
            mask &= mask - 1;
        }
        i += 32;
    }
    while i < len {
        let c = buf[i];
        if c == b',' || c == b'\n' || c == b'"' {
            out.push(i as u32);
        }
        i += 1;
    }
}

/// Returns the index of the next `,`, `\n`, or `"` at or after `start`.
/// Returns `buf.len()` if none remain.
#[inline(always)]
pub fn next_structural(buf: &[u8], start: usize) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        // SAFETY: this arm is compiled only when AVX2 is a crate target feature,
        // so the CPU running the binary supports the AVX2 function below.
        next_structural_avx2(buf, start)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        next_structural_scalar(buf, start)
    }
}

/// Plain byte scan. Also the tail of the AVX2 loop (a remainder under 32 bytes).
#[inline(always)]
fn next_structural_scalar(buf: &[u8], start: usize) -> usize {
    let len = buf.len();
    let mut i = start;
    while i < len {
        let b = buf[i];
        if b == b',' || b == b'\n' || b == b'"' {
            return i;
        }
        i += 1;
    }
    len
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[target_feature(enable = "avx2")]
fn next_structural_avx2(buf: &[u8], start: usize) -> usize {
    use std::arch::x86_64::{
        _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_or_si256,
        _mm256_set1_epi8,
    };

    let len = buf.len();
    // `len - start` is only evaluated when `start < len` (`||` short-circuits).
    if start >= len || len - start < 32 {
        return next_structural_scalar(buf, start);
    }

    let ptr = buf.as_ptr();
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let quote = _mm256_set1_epi8(b'"' as i8);
    let mut i = start;
    while len - i >= 32 {
        // SAFETY: `len - i >= 32`, so the unaligned 32-byte load stays inside `buf`.
        let v = unsafe { _mm256_loadu_si256(ptr.add(i).cast()) };
        let eq = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(v, comma), _mm256_cmpeq_epi8(v, nl)),
            _mm256_cmpeq_epi8(v, quote),
        );
        let mask = _mm256_movemask_epi8(eq) as u32;
        if mask != 0 {
            return i + mask.trailing_zeros() as usize;
        }
        i += 32;
    }
    next_structural_scalar(buf, i)
}

#[cfg(test)]
mod tests {
    use super::{next_structural, push_structurals};

    #[test]
    fn index_matches_scalar_scan() {
        let buf = b"e1,\"u2\",2026-01-01T00:00:00Z,view,p1,\nabc\n\"x\",y\n";
        let mut idx = Vec::new();
        push_structurals(buf, &mut idx);
        let mut expect = Vec::new();
        for (i, &c) in buf.iter().enumerate() {
            if c == b',' || c == b'\n' || c == b'"' {
                expect.push(i as u32);
            }
        }
        assert_eq!(idx, expect);
        let mut wide = vec![b'a'; 100];
        wide[0] = b',';
        wide[31] = b'"';
        wide[32] = b'\n';
        wide[99] = b',';
        idx.clear();
        push_structurals(&wide, &mut idx);
        assert_eq!(idx, vec![0, 31, 32, 99]);
        // `-` sits next to a comma. The SWAR subtract trick lights that byte up;
        // the confirmed byte must not be reported.
        let trap = b"p35481,-5\neak4mew";
        idx.clear();
        push_structurals(trap, &mut idx);
        let expect: Vec<u32> = trap
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == b',' || c == b'\n' || c == b'"')
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(idx, expect);
    }

    #[test]
    fn empty_buffer() {
        assert_eq!(next_structural(b"", 0), 0);
        assert_eq!(next_structural(b"", 4), 0);
    }

    #[test]
    fn no_structural_bytes() {
        let buf = b"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG";
        assert!(buf.len() > 32);
        assert_eq!(next_structural(buf, 0), buf.len());
        assert_eq!(next_structural(buf, 3), buf.len());
        assert_eq!(next_structural(buf, buf.len()), buf.len());
    }

    #[test]
    fn structural_at_positions() {
        let needles = [b',', b'\n', b'"'];
        let positions = [0usize, 1, 31, 32, 100];
        for needle in needles {
            for idx in positions {
                let mut buf = vec![b'a'; idx + 1];
                buf[idx] = needle;
                assert_eq!(next_structural(&buf, 0), idx, "byte {needle:#x} at {idx}");
                assert_eq!(next_structural(&buf, idx), idx, "start at match {idx}");
                if idx > 0 {
                    buf[0] = b',';
                    assert_eq!(
                        next_structural(&buf, 1),
                        idx,
                        "skip earlier byte, needle {needle:#x} at {idx}"
                    );
                }
                assert_eq!(next_structural(&buf, idx + 1), buf.len());
            }
        }
    }

    #[test]
    fn quote_before_comma_is_found_first() {
        let buf = b"abc\"def,ghi";
        assert_eq!(next_structural(buf, 0), 3);
        assert_eq!(buf[3], b'"');
        assert!(buf[3..].contains(&b','));

        // Quote in the last byte of a 32-byte chunk, comma in the next.
        let mut boundary = vec![b'a'; 64];
        boundary[31] = b'"';
        boundary[32] = b',';
        assert_eq!(next_structural(&boundary, 0), 31);
    }
}
