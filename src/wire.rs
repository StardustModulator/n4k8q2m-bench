//! One-pass scanner for the deliberately small CSV dialect used by the input
//! files.
//!
//! A comma always separates fields and only LF separates records. Quotes are
//! valid only when exactly two of them wrap an entire field; they do not turn
//! commas or LF into data. CR and every non-structural byte are ordinary field
//! bytes.
//!
//! Each 64-byte block becomes three bitmasks (comma, LF, quote); quotes are
//! never visited one by one. The window scanner checks quotes per block with
//! bit arithmetic (see `rows_window`). The simple scanner counts them per row:
//! a row is well quoted exactly when its total quote count equals twice the
//! number of fields that both begin and end with a quote, since each such field
//! holds at least two quotes, so equality forces two per wrapped field and
//! none anywhere else.

/// Call `f` for every well-formed record containing exactly `N` fields.
///
/// The input must begin at a record boundary. It may end either after LF or at
/// the end of a final unterminated record. Empty fields are returned as empty
/// slices, and syntactic quotes are removed from returned quoted fields.
#[inline]
pub(super) fn rows<'b, const N: usize>(buf: &'b [u8], f: impl FnMut([&'b [u8]; N])) {
    rows_window::<N>(buf, WINDOW, f);
}

/// Bytes per two-stage window. A multiple of 64 so only the input's final
/// block is ever partial, and small enough for u16 window offsets.
const WINDOW: usize = 1 << 15;
/// Set on a line entry whose record must take the exact path.
const POISON: u32 = 1 << 31;
/// Written past the last separator: equals no window offset.
const NONE: u16 = u16::MAX;

/// Two stages per window. Stage 1 turns 64-byte blocks into a dense list of
/// every separator's offset (commas and LFs alike), a list of LF offsets, and
/// a quote verdict, with no data-dependent loop in the common case. Stage 2
/// walks records: a record has `N` fields exactly when its `N`th separator is
/// its own LF, so one comparison checks the field count.
///
/// Quotes: a quote whose neighbours are both non-separators lies strictly
/// inside a field ("stray") and always invalidates its record. Every other
/// quote is a field's first or last byte (an "edge" quote), so a field holds
/// zero, one or two of them, and is well quoted exactly when it holds an even
/// number: two edge quotes of one field are distinct bytes only when it has
/// two or more. Hence a record without stray quotes is well quoted exactly when
/// the running parity of edge quotes is even at each of its separators. Stage
/// 1 checks that per block; a block that fails poisons each record it touches,
/// and poisoned records take the exact per-field path. Every other record has
/// valid quotes, so its field is quoted exactly when its last byte is a quote.
#[inline]
pub(super) fn rows_window<'b, const N: usize>(buf: &'b [u8], window: usize, mut f: impl FnMut([&'b [u8]; N])) {
    debug_assert!(window % 64 == 0 && window > 0 && window <= WINDOW);
    let len = buf.len();
    // Slack for the unconditional writes, the EOF entry and the sentinels.
    let mut stops: Vec<u16> = Vec::with_capacity(window + 80);
    let mut lines: Vec<u32> = Vec::with_capacity(window + 80);
    let mut start = 0;
    while start < len {
        let end = (start + window).min(len);
        let next = window_rows::<N>(buf, start, end, stops.as_mut_ptr(), lines.as_mut_ptr(), &mut f);
        if next > start {
            start = next;
        } else {
            // A full window without LF: one long row. Hand it to the simple
            // scanner, which has no length limit.
            let stop = memchr::memchr(b'\n', &buf[end..]).map_or(len, |i| end + i + 1);
            rows_simple::<N>(&buf[start..stop], &mut f);
            start = stop;
        }
    }
}

#[inline(always)]
fn is_sep(b: u8) -> u64 {
    u64::from(b == b',' || b == b'\n')
}

/// Bit i is the XOR of bits 0..=i.
#[inline(always)]
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Write `rel + i` for every set bit i of `m`, ascending, to `dst`. Eight
/// slots are written unconditionally; those past the count hold garbage.
///
/// # Safety
/// `dst` must be valid for `max(8, popcount(m))` writes.
#[inline(always)]
unsafe fn flatten(dst: *mut u16, rel: u32, m: u64) {
    // SAFETY: forwarded.
    #[cfg(target_arch = "aarch64")]
    unsafe { flatten_rev(dst, rel, m) }
    // SAFETY: forwarded.
    #[cfg(not(target_arch = "aarch64"))]
    unsafe { flatten_tz(dst, rel, m) }
}

/// For targets with a trailing-zero count and BLSR.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
unsafe fn flatten_tz(dst: *mut u16, rel: u32, mut m: u64) {
    // SAFETY: the caller guarantees room for max(8, popcount(m)) slots.
    unsafe {
        for j in 0..8 {
            *dst.add(j) = (rel + m.trailing_zeros()) as u16;
            m &= m.wrapping_sub(1);
        }
        let mut j = 8;
        while m != 0 {
            *dst.add(j) = (rel + m.trailing_zeros()) as u16;
            m &= m - 1;
            j += 1;
        }
    }
}

/// AArch64 counts only leading zeros: reverse once, then peel the top bit.
/// An empty word yields 64, and the wrapping shift then clears nothing.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
unsafe fn flatten_rev(dst: *mut u16, rel: u32, m: u64) {
    const TOP: u64 = 1 << 63;
    let mut r = m.reverse_bits();
    // SAFETY: the caller guarantees room for max(8, popcount(m)) slots.
    unsafe {
        for j in 0..8 {
            let b = r.leading_zeros();
            *dst.add(j) = (rel + b) as u16;
            r &= !TOP.wrapping_shr(b);
        }
        let mut j = 8;
        while r != 0 {
            let b = r.leading_zeros();
            *dst.add(j) = (rel + b) as u16;
            r &= !(TOP >> b);
            j += 1;
        }
    }
}

/// Scan `buf[start..end]` and emit every complete row in it. Returns the
/// offset just past the last complete row, or `start` if there was none.
#[inline(always)]
fn window_rows<'b, const N: usize>(
    buf: &'b [u8],
    start: usize,
    end: usize,
    sp: *mut u16,
    lp: *mut u32,
    f: &mut impl FnMut([&'b [u8]; N]),
) -> usize {
    let len = buf.len();
    let (mut nc, mut nl) = (0usize, 0usize);
    let mut prev_s = 1u64; // a window starts at a record boundary
    let mut odd = 0u64; // all ones after an odd count of edge quotes since the last LF
    // The buffer's first record takes the exact path: if its first field is
    // empty, no byte precedes it for the fast path's quote probe.
    let mut pending = if start == 0 { POISON } else { 0 }; // poison for the next LF
    let mut off = start;
    while off < end {
        let rem = end - off;
        let rel = (off - start) as u32;
        // SAFETY: a full block lies inside `buf`; a partial one (only at EOF)
        // is copied into zero padding, which holds no structural byte.
        let (c, n, q) = unsafe {
            if rem >= 64 {
                masks64(buf.as_ptr().add(off))
            } else {
                let mut tmp = [0u8; 64];
                tmp[..rem].copy_from_slice(&buf[off..end]);
                masks64(tmp.as_ptr())
            }
        };
        let s = c | n;
        // The byte after this block, for quotes in its last lane. EOF counts as
        // a separator; in a partial block it sits at bit `rem`.
        let (next_s, tail) = if rem > 64 {
            (is_sep(buf[off + 64]), 0)
        } else if rem == 64 {
            (if off + 64 < len { is_sep(buf[off + 64]) } else { 1 }, 0)
        } else {
            (0, 1u64 << rem)
        };
        let near = (s << 1) | prev_s | ((s | tail) >> 1) | (next_s << 63);
        prev_s = s >> 63;
        let edge = q & near;
        let par = prefix_xor(edge) ^ odd;
        let bad = (q & !near) | (par & s);

        // SAFETY: capacity is window + 80 and a window has at most `window`
        // separators, so the unconditional writes stay in bounds.
        unsafe { flatten(sp.add(nc), rel, s) };

        // LFs: two unconditional entries, then any extras.
        let lfs = n.count_ones() as usize;
        let first = pending & if n != 0 { POISON } else { 0 };
        pending &= if n != 0 { 0 } else { POISON };
        let n1 = n & n.wrapping_sub(1);
        // SAFETY: as for separators; a window has at most `window` LFs.
        unsafe {
            *lp.add(nl) = (rel + n.trailing_zeros()) | first;
            *lp.add(nl + 1) = rel + n1.trailing_zeros();
            if lfs > 2 {
                let mut m = n1 & n1.wrapping_sub(1);
                let mut j = 2;
                while m != 0 {
                    *lp.add(nl + j) = rel + m.trailing_zeros();
                    m &= m - 1;
                    j += 1;
                }
            }
        }
        if bad != 0 {
            (pending, odd) = poison_block(lp, nl, lfs, n, edge, odd);
        } else {
            // No LF here had odd parity, so the parity restarted cleanly at
            // each one and the top bit is exact.
            odd = ((par as i64) >> 63) as u64;
        }
        nc += s.count_ones() as usize;
        nl += lfs;
        off += 64;
    }
    if end == len && buf[len - 1] != b'\n' {
        // EOF terminates an unterminated final record.
        let eof = (len - start) as u32;
        // SAFETY: at most `window` separators and LFs were counted.
        unsafe {
            *sp.add(nc) = eof as u16;
            *lp.add(nl) = eof | pending | if odd != 0 { POISON } else { 0 };
        }
        nc += 1;
        nl += 1;
    }
    // SAFETY: nc + N <= window + 1 + 8 is within capacity. Stage 2 reads up to
    // N - 1 slots past a record's first separator, which are now initialized.
    for j in 0..N {
        unsafe { *sp.add(nc + j) = NONE };
    }
    // SAFETY: entries 0..nl were initialized above.
    let lines = unsafe { std::slice::from_raw_parts(lp, nl) };
    // SAFETY: start < len.
    let base = unsafe { buf.as_ptr().add(start) };

    let mut row = 0usize; // window offset of the record's first byte
    let mut k = 0usize; // index of the record's first separator
    for &line in lines {
        let lf = (line & !POISON) as usize;
        // SAFETY: k <= nc (each record's separators end at its own LF, which
        // is a separator), so k + N - 1 < nc + N is initialized.
        let fast = unsafe { *sp.add(k + N - 1) } as usize == lf && line & POISON == 0;
        if fast {
            let mut fields = [&buf[..0]; N];
            let mut s = row;
            for (i, field) in fields.iter_mut().enumerate() {
                // SAFETY: k + i < nc, and s <= e <= len - start. The probe
                // reads the byte before the separator: the field's last byte,
                // or for an empty field the previous separator. The record
                // before it ends with LF unless this is the buffer's first
                // record, which is poisoned, so e - 1 >= -start.
                let e = unsafe { *sp.add(k + i) } as usize;
                let w = (unsafe { *base.add(e).sub(1) } == b'"') as usize;
                // SAFETY: a valid field ending in a quote is wrapped and has
                // two or more bytes.
                *field = unsafe { std::slice::from_raw_parts(base.add(s + w), e - s - 2 * w) };
                s = e + 1;
            }
            f(fields);
            k += N;
        } else {
            // SAFETY: every LF entry is also a separator entry at or after k.
            let mut j = k;
            while unsafe { *sp.add(j) } as usize != lf {
                j += 1;
            }
            if j - k == N - 1 {
                // SAFETY: entries k..=j were initialized in stage 1.
                let ends = unsafe { std::slice::from_raw_parts(sp.add(k), N) };
                exact::<N>(buf, start, start + row, ends, f);
            }
            k = j + 1;
        }
        row = lf + 1;
    }
    (start + row).min(len)
}

/// Check a poisoned record byte by byte and emit it if well formed. `ends`
/// holds its N separator offsets relative to `start`.
#[cold]
#[inline(never)]
fn exact<'b, const N: usize>(buf: &'b [u8], start: usize, row: usize, ends: &[u16], f: &mut impl FnMut([&'b [u8]; N])) {
    let mut fields = [&buf[..0]; N];
    let mut s = row;
    for (field, &e) in fields.iter_mut().zip(ends) {
        let e = start + e as usize;
        let raw = &buf[s..e];
        *field = match raw.iter().filter(|&&b| b == b'"').count() {
            0 => raw,
            2 if raw.len() >= 2 && raw[0] == b'"' && raw[raw.len() - 1] == b'"' => &raw[1..raw.len() - 1],
            _ => return,
        };
        s = e + 1;
    }
    f(fields);
}

/// A block failed the quote check: poison its LFs and the record continuing
/// past it. The parity carried out only matters for that poisoned record: an
/// unpoisoned record always starts at an LF whose parity was checked even. It
/// is recomputed exactly anyway so a quoted field spanning the next block
/// boundary does not poison that block too.
#[cold]
#[inline(never)]
fn poison_block(lp: *mut u32, nl: usize, lfs: usize, n: u64, edge: u64, odd: u64) -> (u32, u64) {
    for j in 0..lfs {
        // SAFETY: entries nl..nl + lfs were written for this block's LFs.
        unsafe { *lp.add(nl + j) |= POISON };
    }
    let flip = if n == 0 { (odd ^ edge.count_ones() as u64) & 1 } else { (((edge >> (63 - n.leading_zeros())) >> 1).count_ones() & 1) as u64 };
    (POISON, 0u64.wrapping_sub(flip))
}

/// The single-stage scanner: no row-length limit.
#[inline]
fn rows_simple<'b, const N: usize>(buf: &'b [u8], mut f: impl FnMut([&'b [u8]; N])) {
    debug_assert!((2..=8).contains(&N));
    let len = buf.len();
    if len == 0 {
        return;
    }
    let mut st = Rows::<N> { ends: [0; 8], cnt: 0, row_start: 0, qrow: 0, qbase: 0 };
    let mut at = 0usize;
    while len - at >= 64 {
        // SAFETY: the loop condition keeps all 64 bytes inside `buf`.
        let (c, n, q) = unsafe { masks64(buf.as_ptr().add(at)) };
        st.block(buf, at, c, n, q, &mut f);
        at += 64;
    }
    let tail_nl = buf[len - 1] != b'\n';
    if at < len {
        // Zero bytes are not structural, so padding adds no bits.
        let mut tmp = [0u8; 64];
        tmp[..len - at].copy_from_slice(&buf[at..]);
        // SAFETY: `tmp` holds exactly 64 bytes.
        let (c, mut n, q) = unsafe { masks64(tmp.as_ptr()) };
        if tail_nl {
            // EOF terminates an unterminated final record.
            n |= 1u64 << (len - at);
        }
        st.block(buf, at, c, n, q, &mut f);
    } else if tail_nl {
        let rowq = st.qbase - st.qrow;
        st.finish(buf, len, rowq, &mut f);
    }
}

struct Rows<const N: usize> {
    ends: [usize; 8],
    cnt: usize,
    row_start: usize,
    qrow: u32,
    qbase: u32,
}

impl<const N: usize> Rows<N> {
    #[inline(always)]
    fn push_commas(&mut self, base: usize, mut m: u64) {
        while m != 0 {
            self.ends[self.cnt & 7] = base + m.trailing_zeros() as usize;
            self.cnt += 1;
            m &= m - 1;
        }
    }

    #[inline(always)]
    fn block<'b>(&mut self, buf: &'b [u8], base: usize, mut c: u64, mut n: u64, q: u64,
                 f: &mut impl FnMut([&'b [u8]; N])) {
        while n != 0 {
            let bit = n.trailing_zeros();
            let below = (1u64 << bit) - 1;
            self.push_commas(base, c & below);
            c &= !below;
            let qnow = self.qbase + (q & below).count_ones();
            self.finish(buf, base + bit as usize, qnow - self.qrow, f);
            self.qrow = qnow;
            n &= n - 1;
        }
        self.push_commas(base, c);
        self.qbase += q.count_ones();
    }

    /// Close the record whose terminator (LF or EOF) is at `end`.
    #[inline(always)]
    fn finish<'b>(&mut self, buf: &'b [u8], end: usize, rowq: u32, f: &mut impl FnMut([&'b [u8]; N])) {
        if self.cnt == N - 1 {
            let last = buf.len() - 1;
            let mut fields = [&buf[..0]; N];
            let mut wrapped = 0u32;
            let mut s = self.row_start;
            for (i, field) in fields.iter_mut().enumerate() {
                let e = if i == N - 1 { end } else { self.ends[i] };
                debug_assert!(s <= e && e <= buf.len());
                // SAFETY: both indexes are clamped into a non-empty `buf`.
                let (first, final_) = unsafe {
                    (*buf.get_unchecked(s.min(last)), *buf.get_unchecked(e.saturating_sub(1)))
                };
                let w = ((e - s >= 2) & (first == b'"') & (final_ == b'"')) as usize;
                // SAFETY: separators are visited in increasing order, so
                // `s <= e`, and `w` is 1 only when the field has 2+ bytes.
                *field = unsafe { buf.get_unchecked(s + w..e - w) };
                wrapped += 2 * w as u32;
                s = e + 1;
            }
            if wrapped == rowq {
                f(fields);
            }
        }
        self.cnt = 0;
        self.row_start = end + 1;
    }
}

/// Comma, LF and quote masks of 64 bytes at `p`; bit i describes `p[i]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    #[cfg(target_feature = "avx2")]
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm256_loadu_si256(p.cast());
        let b = _mm256_loadu_si256(p.add(32).cast());
        let m = |c: u8| {
            let k = _mm256_set1_epi8(c as i8);
            let lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, k)) as u32 as u64;
            let hi = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, k)) as u32 as u64;
            lo | hi << 32
        };
        (m(b','), m(b'\n'), m(b'"'))
    }
    #[cfg(not(target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let v = [0, 16, 32, 48].map(|o| _mm_loadu_si128(p.add(o).cast()));
        let m = |c: u8| {
            let k = _mm_set1_epi8(c as i8);
            v.iter().enumerate().fold(0u64, |acc, (i, &x)| {
                acc | (_mm_movemask_epi8(_mm_cmpeq_epi8(x, k)) as u32 as u64) << (16 * i)
            })
        };
        (m(b','), m(b'\n'), m(b'"'))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    unsafe {
        let w = vld1q_u8(WEIGHTS.as_ptr());
        let v = [vld1q_u8(p), vld1q_u8(p.add(16)), vld1q_u8(p.add(32)), vld1q_u8(p.add(48))];
        let m = |c: u8| {
            let k = vdupq_n_u8(c);
            let s0 = vpaddq_u8(vandq_u8(vceqq_u8(v[0], k), w), vandq_u8(vceqq_u8(v[1], k), w));
            let s1 = vpaddq_u8(vandq_u8(vceqq_u8(v[2], k), w), vandq_u8(vceqq_u8(v[3], k), w));
            let s = vpaddq_u8(s0, s1);
            vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(s, s)), 0)
        };
        (m(b','), m(b'\n'), m(b'"'))
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    let (mut c, mut n, mut q) = (0u64, 0u64, 0u64);
    for i in 0..64 {
        // SAFETY: callers pass 64 readable bytes.
        let b = unsafe { *p.add(i) };
        c |= u64::from(b == b',') << i;
        n |= u64::from(b == b'\n') << i;
        q |= u64::from(b == b'"') << i;
    }
    (c, n, q)
}

#[cfg(test)]
mod tests {
    use super::{rows, rows_simple, rows_window};

    fn reference<'b, const N: usize>(buf: &'b [u8], mut emit: impl FnMut([&'b [u8]; N])) {
        fn line<'b, const N: usize>(line: &'b [u8]) -> Option<[&'b [u8]; N]> {
            let mut fields = [&line[..0]; N];
            let mut count = 0usize;
            for raw in line.split(|&byte| byte == b',') {
                if count == N {
                    return None;
                }
                let quotes = raw.iter().filter(|&&byte| byte == b'"').count();
                fields[count] = match quotes {
                    0 => raw,
                    2 if raw.len() >= 2 && raw[0] == b'"' && raw[raw.len() - 1] == b'"' => {
                        &raw[1..raw.len() - 1]
                    }
                    _ => return None,
                };
                count += 1;
            }
            (count == N).then_some(fields)
        }

        let mut start = 0usize;
        for (at, &byte) in buf.iter().enumerate() {
            if byte == b'\n' {
                if let Some(fields) = line::<N>(&buf[start..at]) {
                    emit(fields);
                }
                start = at + 1;
            }
        }
        if start < buf.len() {
            if let Some(fields) = line::<N>(&buf[start..]) {
                emit(fields);
            }
        }
    }

    fn owned<const N: usize>(buf: &[u8], simd: bool) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut push = |fields: [&[u8]; N]| {
            out.push(fields.iter().map(|field| field.to_vec()).collect());
        };
        if simd {
            rows::<N>(buf, &mut push);
        } else {
            reference::<N>(buf, &mut push);
        }
        out
    }

    fn collect<const N: usize>(buf: &[u8], run: impl FnOnce(&[u8], &mut dyn FnMut([&[u8]; N]))) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        run(buf, &mut |fields: [&[u8]; N]| out.push(fields.iter().map(|field| field.to_vec()).collect()));
        out
    }

    fn check<const N: usize>(buf: &[u8]) {
        let want = owned::<N>(buf, false);
        assert_eq!(owned::<N>(buf, true), want, "input={buf:?}");
        assert_eq!(collect::<N>(buf, |b, f| rows_simple::<N>(b, f)), want, "simple input={buf:?}");
        for window in [64, 128, 192, 256] {
            assert_eq!(collect::<N>(buf, |b, f| rows_window::<N>(b, window, f)), want, "window={window} input={buf:?}");
        }
    }

    #[test]
    fn stray_and_edge_quotes() {
        for input in [
            b"a\"b,c\n".as_slice(), b"\"a\"b,c\n", b"\",c\n", b"\"\",c\n", b"\"\"\",c\n", b"\"\"\"\",c\n",
            b"a,\"\n", b"a,\"b\"\r\n", b"a,\"b\r\"\n", b"\"a\",\"b\"\n", b"\"a,b\"\n", b"a\",\"b\n",
            b"x\nab\"c,d\ne,f\n", b"\"\",\"\"", b"a,\"", b"a,\"\"", b"\"", b",\"\"\n,",
        ] {
            check::<2>(input);
        }
        // A stray quote poisons only its own row, also across 64-byte blocks.
        let mut long = Vec::new();
        for i in 0..40usize {
            long.extend(std::iter::repeat_n(b'x', i));
            long.extend_from_slice(if i % 3 == 0 { b"a\"b,c\n" } else { b"\"q\",zz\n" });
        }
        check::<2>(&long);
    }

    #[test]
    fn exact_dialect_examples() {
        check::<2>(b"a,b\n\"c\",\"\"\nlast,record");
        check::<4>(b"a,b,c,d\n\"a\",b,\"c\",d\n");
        check::<7>(b"e1,u1,2026-01-01T00:00:00Z,2026-01-01T00:00:01Z,view,p1,\n");

        assert_eq!(
            owned::<2>(b"\"hello\",\"world\"\n", true),
            vec![vec![b"hello".to_vec(), b"world".to_vec()]]
        );
        // A quoted slice points into the source and excludes both wrappers.
        let source = b"\"ab\",cd\n";
        rows::<2>(source, |fields| {
            assert_eq!(fields[0].as_ptr(), unsafe { source.as_ptr().add(1) });
            assert_eq!(fields[0], b"ab");
        });
    }

    #[test]
    fn malformed_quotes_and_field_counts_are_dropped() {
        for input in [
            b"\"a,b\n".as_slice(),
            b"a\",b\n",
            b"\"a\"x,b\n",
            b"x\"a\",b\n",
            b"\"a\"\",b\n",
            b"a\n",
            b"a,b,c\n",
            b"\n",
        ] {
            check::<2>(input);
            assert!(owned::<2>(input, true).is_empty(), "input={input:?}");
        }
        // Commas are delimiters even between quotes in this dialect.
        assert!(owned::<2>(b"\"a,b\"\n", true).is_empty());
    }

    #[test]
    fn eof_empty_cr_and_utf8_are_bytes_exact() {
        check::<2>(b"a,b");
        check::<2>(b"a,");
        check::<2>(b",b");
        check::<2>(b",");
        check::<2>(b"a,b\n");
        check::<2>(b"a,b\n\n");
        check::<2>(b"a,b\nlast,");
        check::<2>("é,雪\nλ,🙂".as_bytes());

        assert_eq!(owned::<2>(b"a,b\r\n", true)[0][1], b"b\r");
        // CR after a closing quote means the quote no longer wraps the whole
        // field, because CR is ordinary data rather than part of the newline.
        assert!(owned::<2>(b"a,\"b\"\r\n", true).is_empty());
    }

    #[test]
    fn every_simd_boundary_and_chunk_tail() {
        for len in 0..=96usize {
            let a = vec![b'a'; len];
            let b = vec![b'b'; 97 - len];
            let mut input = Vec::new();
            input.extend_from_slice(&a);
            input.push(b',');
            input.extend_from_slice(&b);
            if len & 1 == 0 {
                input.push(b'\n');
            }
            check::<2>(&input);

            let mut quoted = Vec::new();
            quoted.push(b'"');
            quoted.extend_from_slice(&a);
            quoted.extend_from_slice(b"\",tail\n");
            check::<2>(&quoted);
        }

        // Shift later records through every position modulo 32.
        for shift in 0..64usize {
            let mut input = Vec::new();
            input.extend(std::iter::repeat_n(b'x', shift));
            input.extend_from_slice(b",z\n\"left\",right\n");
            check::<2>(&input);
        }
    }

    /// Mostly valid records with sparse dirt, so the fast path and the
    /// poisoned-block recovery both run often, at every window size.
    fn realistic(next: &mut impl FnMut() -> u64, rows: usize, fields: usize, dirt: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let word = |out: &mut Vec<u8>, next: &mut dyn FnMut() -> u64, n: u64| {
            for _ in 0..next() % n {
                out.push(b"abcxyz0189Q _"[next() as usize % 13]);
            }
        };
        for _ in 0..rows {
            let bad = if next() % 1000 < dirt { 1 + next() % 11 } else { 0 };
            let width = match bad { 1 => 0, 2 => fields + 1, 3 => fields - 1, _ => fields };
            let hit = next() as usize % fields;
            for i in 0..width {
                if i > 0 {
                    out.push(b',');
                }
                match (bad, i == hit) {
                    (4, true) => out.extend_from_slice(b"ab\"cd"),
                    (5, true) => out.extend_from_slice(b"\"ab"),
                    (6, true) => out.extend_from_slice(b"ab\""),
                    (7, true) => out.push(b'"'),
                    (8, true) => out.extend_from_slice(b"\"\"\""),
                    (9, true) => out.extend_from_slice(b"\"a,b\""),
                    (10, true) => word(&mut out, next, 400),
                    _ => match next() % 20 {
                        0 | 1 => {}
                        2..=5 => {
                            out.push(b'"');
                            word(&mut out, next, 13);
                            out.push(b'"');
                        }
                        6 => out.extend_from_slice(b"\"\""),
                        7 => word(&mut out, next, 60),
                        _ => word(&mut out, next, 13),
                    },
                }
            }
            if bad == 11 {
                out.push(b'\r');
            }
            out.push(b'\n');
        }
        if next() % 2 == 0 {
            out.pop();
        }
        out
    }

    #[test]
    fn differential_realistic_rows() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..1500 {
            let fields = [2, 4, 7][round % 3];
            let dirt = [0, 2, 20, 200][round / 3 % 4];
            let rows = 1 + (next() % 160) as usize;
            let input = realistic(&mut next, rows, fields, dirt);
            check::<2>(&input);
            check::<4>(&input);
            check::<7>(&input);
        }
        // Every window may begin with an empty field.
        check::<2>(&b",x\n".repeat(200));
        check::<2>(&b",\"x\"\n".repeat(200));
        check::<3>(&b",,\n".repeat(300));
    }

    #[test]
    fn flatten_variants_agree() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut masks = vec![0, 1, 1 << 63, u64::MAX, 0x8000_0000_0000_0001, 0x5555_5555_5555_5555];
        for _ in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            masks.push(state & state.rotate_left(17) & state.rotate_left(31));
            masks.push(state);
        }
        for m in masks {
            let (mut a, mut b) = ([0u16; 72], [0u16; 72]);
            // SAFETY: 72 slots cover max(8, popcount) for any mask.
            unsafe {
                super::flatten_tz(a.as_mut_ptr(), 4096, m);
                super::flatten_rev(b.as_mut_ptr(), 4096, m);
            }
            let want: Vec<u16> = (0..64).filter(|i| m >> i & 1 == 1).map(|i| 4096 + i as u16).collect();
            assert_eq!(&a[..want.len()], &want[..], "tz m={m:#x}");
            assert_eq!(&b[..want.len()], &want[..], "rev m={m:#x}");
        }
    }

    #[test]
    fn differential_random_bytes() {
        let alphabet = [
            b',', b'\n', b'"', b'\r', b'a', b'Z', b'0', b' ', 0, 0x7f, 0x80, 0xa9, 0xc3, 0xe9, 0xff,
        ];
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for _ in 0..12_000 {
            let len = (next() as usize) & 511;
            let mut input = Vec::with_capacity(len);
            for _ in 0..len {
                input.push(alphabet[next() as usize % alphabet.len()]);
            }
            check::<2>(&input);
            check::<4>(&input);
            check::<7>(&input);
        }
    }
}

#[cfg(test)]
pub(super) unsafe fn masks64_for_bench(p: *const u8) -> (u64, u64, u64) {
    unsafe { masks64(p) }
}
