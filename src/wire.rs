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
//! bit arithmetic (see `cells_window`). The simple scanner counts them per row:
//! a row is well quoted exactly when its total quote count equals twice the
//! number of fields that both begin and end with a quote, since each such field
//! holds at least two quotes, so equality forces two per wrapped field and
//! none anywhere else.
//!
//! Well-formed records are handed over as `Cell`s: a base pointer and the
//! offsets of the record's separators. Slicing a field costs a few
//! instructions, so a decoder slices only the fields it reads, and the hot
//! loop carries one 40-byte cell per record instead of `N` fat slices.

use std::marker::PhantomData;

/// A well-formed record of `N` fields: `off[i]` is the offset from `base` of
/// the separator ending field `i` (the last one is its LF, or EOF), and
/// `off[7]` is the offset of its first byte. Every field has either no quote
/// or exactly two wrapping it, so a field is quoted exactly when the byte
/// before its separator is a quote.
#[derive(Clone, Copy)]
pub(super) struct Cell<'b, const N: usize> {
    base: *const u8,
    off: [u32; 8],
    _buf: PhantomData<&'b [u8]>,
}

impl<'b, const N: usize> Cell<'b, N> {
    /// A placeholder for batch arrays; never read as a record.
    pub(super) const EMPTY: Self = Self { base: std::ptr::null(), off: [0; 8], _buf: PhantomData };

    /// # Safety
    /// `off[7] <= off[0] <= ... <= off[N-1]` index `base`'s buffer, and the
    /// byte before every separator is readable (a record never starts at
    /// the buffer's first byte).
    #[inline(always)]
    unsafe fn new(base: *const u8, off: [u32; 8]) -> Self {
        debug_assert!(N <= 7 && (0..N).all(|i| if i == 0 { off[7] <= off[0] } else { off[i - 1] < off[i] }));
        Self { base, off, _buf: PhantomData }
    }

    /// Field `i` with its wrapping quotes, if any, removed.
    #[inline(always)]
    pub(super) fn field(&self, i: usize) -> &'b [u8] {
        debug_assert!(i < N);
        let s = if i == 0 { self.off[7] } else { self.off[i - 1] + 1 } as usize;
        let e = self.off[i] as usize;
        // SAFETY: by `new`, s <= e index the buffer and base[e - 1] is
        // readable: the field's last byte, or the previous separator (never
        // a quote) when the field is empty. A field ending in a quote is
        // wrapped, so it has two or more bytes.
        unsafe {
            let w = usize::from(*self.base.add(e).sub(1) == b'"');
            std::slice::from_raw_parts(self.base.add(s + w), e - s - 2 * w)
        }
    }

    #[inline(always)]
    pub(super) fn fields(&self) -> [&'b [u8]; N] {
        let mut fields = [&[][..]; N];
        for (i, f) in fields.iter_mut().enumerate() {
            *f = self.field(i);
        }
        fields
    }

}

/// Call `f` for every well-formed record containing exactly `N` fields.
///
/// The input must begin at a record boundary. It may end either after LF or at
/// the end of a final unterminated record. Empty fields are returned as empty
/// slices, and syntactic quotes are removed from returned quoted fields.
#[inline]
pub(super) fn rows<'b, const N: usize>(buf: &'b [u8], mut f: impl FnMut([&'b [u8]; N])) {
    // The first record may start at byte 0, which cells never do.
    let start = memchr::memchr(b'\n', buf).map_or(buf.len(), |i| i + 1);
    if let Some(fields) = first_record::<N>(&buf[..start]) {
        f(fields);
    }
    cells::<N>(buf, start, buf.len(), |c| f(c.fields()));
}

/// Emit the well-formed `N`-field records of `buf[start..end]`. `start` is a
/// record boundary at or after byte 1, so a separator is always preceded by
/// a readable byte. `end` is `buf.len()` or the offset just past an LF.
#[inline]
pub(super) fn cells<'b, const N: usize>(buf: &'b [u8], start: usize, end: usize, f: impl FnMut(Cell<'b, N>)) {
    cells_window::<N>(buf, start, end, WINDOW, f);
}

/// The record `line` (without its LF), split and checked byte by byte.
#[cold]
#[inline(never)]
fn first_record<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let mut fields = [&line[..0]; N];
    let mut count = 0usize;
    for raw in line.split(|&b| b == b',') {
        if count == N {
            return None;
        }
        fields[count] = match raw.iter().filter(|&&b| b == b'"').count() {
            0 => raw,
            2 if raw.len() >= 2 && raw[0] == b'"' && raw[raw.len() - 1] == b'"' => &raw[1..raw.len() - 1],
            _ => return None,
        };
        count += 1;
    }
    (count == N && !line.is_empty()).then_some(fields)
}

/// Bytes per two-stage window. A multiple of 64 so only the input's final
/// block is ever partial, and small enough for u16 window offsets. Stage 2
/// rereads the window, so it and its lists (about 10 KiB) should stay in L1
/// next to the caller's own data; on M1 8 to 32 KiB measure the same.
const WINDOW: usize = 1 << 13;
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
/// quote opens its field (it follows a separator), closes it (a separator
/// follows it), or both, which makes the field a lone quote: invalid. A field
/// is well quoted exactly when it has an opener and a closer or neither.
///
/// Stage 1 checks this per block by subtraction. `C - O - borrow` (closers
/// minus openers as integers) borrows from an opener up to the next closer,
/// so bit i of the difference is set exactly when byte i lies inside an open
/// quote, except that an opener or a closer reached outside one also yields a
/// set bit. A record whose quote state starts closed is then well quoted
/// exactly when no separator and no closer of it has its bit set: the first
/// opener left open sets the bit at its field's end, and a closer without an
/// opener sets its own. A block that fails poisons every record it touches,
/// and poisoned records take the exact per-field path. The state carried into
/// a record is wrong only after a record left a quote open, and then the LF
/// before it is set too, so its block fails and the record is poisoned.
/// Every other record has valid quotes, so its field is quoted exactly when
/// its last byte is a quote.
#[inline]
pub(super) fn cells_window<'b, const N: usize>(buf: &'b [u8], mut start: usize, len: usize, window: usize,
                                                mut f: impl FnMut(Cell<'b, N>)) {
    debug_assert!(window % 64 == 0 && window > 0 && window <= WINDOW);
    if start >= len {
        return;
    }
    assert!(N <= 7 && start >= 1 && len <= buf.len());
    debug_assert!(buf[start - 1] == b'\n');
    // Zeroed, with slack for the unconditional writes, the EOF entry and the
    // sentinels: stage 2 copies eight slots per record, some past its own.
    let mut stops: Vec<u16> = vec![0; window + 80];
    let mut lines: Vec<u32> = vec![0; window + 80];
    while start < len {
        let end = (start + window).min(len);
        let next = window_rows::<N>(buf, start, end, len, stops.as_mut_ptr(), lines.as_mut_ptr(), &mut f);
        if next > start {
            start = next;
        } else {
            // A full window without LF: one long row. Hand it to the simple
            // scanner, which has no length limit.
            let stop = memchr::memchr(b'\n', &buf[end..len]).map_or(len, |i| end + i + 1);
            rows_simple::<N>(buf, start, stop, &mut f);
            start = stop;
        }
    }
}

/// 1 for a separator byte: one load instead of two compares per block.
static SEP: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b',' as usize] = 1;
    t[b'\n' as usize] = 1;
    t
};

#[inline(always)]
fn is_sep(b: u8) -> u64 {
    u64::from(SEP[b as usize])
}

/// A window offset as stored in a list.
trait Slot: Copy {
    fn at(v: u32) -> Self;
}
impl Slot for u16 {
    #[inline(always)]
    fn at(v: u32) -> u16 {
        v as u16
    }
}
impl Slot for u32 {
    #[inline(always)]
    fn at(v: u32) -> u32 {
        v
    }
}

/// Write `rel + i` for every set bit i of `m`, ascending, to `dst`. `k` slots
/// (a constant after inlining) are written unconditionally; those past the
/// count hold garbage. `k` covers almost every block of the input's row
/// shapes, so the loop rarely runs.
///
/// # Safety
/// `dst` must be valid for `max(k, popcount(m))` writes.
#[inline(always)]
unsafe fn flatten<T: Slot>(dst: *mut T, rel: u32, m: u64, k: usize) {
    // SAFETY: forwarded.
    #[cfg(target_arch = "aarch64")]
    unsafe { flatten_rev(dst, rel, m, k) }
    // SAFETY: forwarded.
    #[cfg(not(target_arch = "aarch64"))]
    unsafe { flatten_tz(dst, rel, m, k) }
}

/// For targets with a trailing-zero count and BLSR.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
unsafe fn flatten_tz<T: Slot>(dst: *mut T, rel: u32, mut m: u64, k: usize) {
    // SAFETY: the caller guarantees room for max(k, popcount(m)) slots.
    unsafe {
        for j in 0..k {
            *dst.add(j) = T::at(rel + m.trailing_zeros());
            m &= m.wrapping_sub(1);
        }
        let mut j = k;
        while m != 0 {
            *dst.add(j) = T::at(rel + m.trailing_zeros());
            m &= m - 1;
            j += 1;
        }
    }
}

/// AArch64 counts only leading zeros: reverse once, then peel the top bit.
/// An empty word yields 64, and the wrapping shift then clears nothing.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
unsafe fn flatten_rev<T: Slot>(dst: *mut T, rel: u32, m: u64, k: usize) {
    const TOP: u64 = 1 << 63;
    let mut r = m.reverse_bits();
    // SAFETY: the caller guarantees room for max(k, popcount(m)) slots.
    unsafe {
        for j in 0..k {
            let b = r.leading_zeros();
            *dst.add(j) = T::at(rel + b);
            r &= !TOP.wrapping_shr(b);
        }
        let mut j = k;
        while r != 0 {
            let b = r.leading_zeros();
            *dst.add(j) = T::at(rel + b);
            r &= !(TOP >> b);
            j += 1;
        }
    }
}

/// Stage 1 state carried from block to block.
struct Scan {
    sp: *mut u16, // next separator slot
    lp: *mut u32, // next LF slot
    prev_s: u64,  // bit 0: the byte before this block is a separator
    open: u64,    // 1 inside an open quote: the borrow out of the last block
    pending: u32, // poison for the next LF
}

impl Scan {
    /// List the separators and LFs of the block at window offset `rel` and
    /// check its quotes. `next_s` says whether the byte after it is a
    /// separator; `tail` marks EOF inside a partial block.
    ///
    /// # Safety
    /// The slots need room for this block's entries plus the slack of the
    /// unconditional writes.
    #[inline(always)]
    unsafe fn block<const N: usize>(&mut self, rel: u32, (c, n, q): (u64, u64, u64), next_s: u64, tail: u64) {
        let s = c | n;
        let o = q & ((s << 1) | self.prev_s);
        let z = q & (((s | tail) >> 1) | (next_s << 63));
        self.prev_s = s >> 63;
        // O + open cannot overflow: an opener at every bit needs separators.
        let (inside, borrow) = z.overflowing_sub(o + self.open);
        self.open = borrow as u64;
        let bad = (inside & (s | z)) | (q & !(o | z)) | (o & z);
        let lfs = n.count_ones() as usize;
        // SAFETY: forwarded from the caller.
        unsafe {
            flatten(self.sp, rel, s, 7);
            flatten(self.lp, rel, n, lf_slots(N));
            if self.pending != 0 && n != 0 {
                *self.lp |= POISON;
                self.pending = 0;
            }
            if bad != 0 {
                self.pending = poison_block(self.lp, lfs);
            }
            self.sp = self.sp.add(s.count_ones() as usize);
            self.lp = self.lp.add(lfs);
        }
    }
}

/// Unconditional LF slots per block: rows of N fields are shorter for small N.
const fn lf_slots(n: usize) -> usize {
    if n >= 6 { 1 } else if n >= 3 { 2 } else { 4 }
}

/// Scan `buf[start..end]` and emit every complete row in it; `len` is where
/// the records end (EOF). Returns the offset just past the last complete row,
/// or `start` if there was none.
#[inline(always)]
fn window_rows<'b, const N: usize>(
    buf: &'b [u8],
    start: usize,
    end: usize,
    len: usize,
    sp: *mut u16,
    lp: *mut u32,
    f: &mut impl FnMut(Cell<'b, N>),
) -> usize {
    // A window starts at a record boundary.
    let mut st = Scan { sp, lp, prev_s: 1, open: 0, pending: 0 };
    // Blocks followed by a byte of the records. At EOF the final block, full
    // or partial, is left for after the loop.
    let inner = if end < len { end } else { start + (len - start - 1) / 64 * 64 };
    // SAFETY: start <= inner <= len <= buf.len().
    let (base, stop) = unsafe { (buf.as_ptr().add(start), buf.as_ptr().add(inner)) };
    let mut p = base;
    let mut rel = 0u32;
    while p < stop {
        // SAFETY: p + 64 <= stop < len (stop is the window end before EOF, or
        // the final block's start), so the block and the byte after it lie
        // inside `buf`. Capacity covers every slot (see cells_window).
        unsafe {
            let m = masks64(p);
            let next_s = is_sep(*p.add(64));
            st.block::<N>(rel, m, next_s, 0);
            p = p.add(64);
        }
        rel += 64;
    }
    if end == len {
        // EOF is a separator after the final block, inside it if partial.
        let off = start + rel as usize;
        let rem = len - off;
        let mut tmp = [0u8; 64];
        tmp[..rem].copy_from_slice(&buf[off..len]);
        let (next_s, tail) = if rem == 64 { (1, 0) } else { (0, 1u64 << rem) };
        // SAFETY: `tmp` holds 64 bytes, zero padded: no structural byte.
        unsafe { st.block::<N>(rel, masks64(tmp.as_ptr()), next_s, tail) };
        if buf[len - 1] != b'\n' {
            // EOF terminates an unterminated final record.
            let eof = (len - start) as u32;
            // SAFETY: at most `window` separators and LFs were listed.
            unsafe {
                *st.sp = eof as u16;
                *st.lp = eof | st.pending | if st.open != 0 { POISON } else { 0 };
                st.sp = st.sp.add(1);
                st.lp = st.lp.add(1);
            }
        }
    }
    // SAFETY: both cursors only moved forward from their bases.
    let (nc, nl) = unsafe { (st.sp.offset_from(sp) as usize, st.lp.offset_from(lp) as usize) };
    // SAFETY: nc + 8 <= window + 1 + 8 is within capacity. Stage 2 reads up to
    // 7 slots past a record's first separator, which are now initialized.
    for j in 0..8 {
        unsafe { *sp.add(nc + j) = NONE };
    }
    // SAFETY: entries 0..nl were initialized above.
    let lines = unsafe { std::slice::from_raw_parts(lp, nl) };

    let mut row = 0usize; // window offset of the record's first byte
    let mut k = 0usize; // index of the record's first separator
    for &line in lines {
        let lf = (line & !POISON) as usize;
        // SAFETY: k <= nc (each record's separators end at its own LF, which
        // is a separator), so k + 7 < nc + 8 is initialized.
        let fast = unsafe { *sp.add(k + N - 1) } as usize == lf && line & POISON == 0;
        if fast {
            // SAFETY: as above; `sp` is 2-aligned, so the eight-slot read is
            // an aligned [u16; 8] read.
            let ends = unsafe { *sp.add(k).cast::<[u16; 8]>() };
            let mut off = ends.map(u32::from);
            off[7] = row as u32;
            // SAFETY: the record's separators ascend from `row` to its LF
            // within the window, and base[-1] is the LF before the window
            // (start >= 1) or a byte of the window.
            f(unsafe { Cell::new(base, off) });
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
                exact::<N>(buf, start, row, ends, f);
            }
            k = j + 1;
        }
        row = lf + 1;
    }
    (start + row).min(len)
}

/// Check a poisoned record byte by byte and emit it if well formed. `row`
/// and `ends` (its N separator offsets) are relative to `start`.
#[cold]
#[inline(never)]
fn exact<'b, const N: usize>(buf: &'b [u8], start: usize, row: usize, ends: &[u16], f: &mut impl FnMut(Cell<'b, N>)) {
    let mut s = start + row;
    let mut off = [0u32; 8];
    for (i, &e) in ends.iter().enumerate() {
        let e = start + e as usize;
        let raw = &buf[s..e];
        let quotes = raw.iter().filter(|&&b| b == b'"').count();
        if !(quotes == 0 || (quotes == 2 && raw.len() >= 2 && raw[0] == b'"' && raw[raw.len() - 1] == b'"')) {
            return;
        }
        off[i] = e as u32;
        s = e + 1;
    }
    off[7] = (start + row) as u32;
    // SAFETY: offsets ascend from the record's first byte, which is at or
    // after byte 1 of `buf`, to its LF or EOF, all inside `buf`.
    f(unsafe { Cell::new(buf.as_ptr(), off) });
}

/// A block failed the quote check: poison its LFs, and return the poison for
/// the record continuing past it.
#[cold]
#[inline(never)]
fn poison_block(lp: *mut u32, lfs: usize) -> u32 {
    for j in 0..lfs {
        // SAFETY: lfs entries at lp were written for this block's LFs.
        unsafe { *lp.add(j) |= POISON };
    }
    POISON
}

/// The single-stage scanner: no row-length limit. Scans `buf[start..end]`,
/// `start >= 1` being a record boundary.
#[inline]
fn rows_simple<'b, const N: usize>(buf: &'b [u8], start: usize, end: usize, mut f: impl FnMut(Cell<'b, N>)) {
    debug_assert!((2..=7).contains(&N) && start >= 1 && end <= buf.len());
    if start >= end {
        return;
    }
    let mut st = Rows::<N> { ends: [0; 8], cnt: 0, row_start: start, qrow: 0, qbase: 0 };
    let mut at = start;
    while end - at >= 64 {
        // SAFETY: the loop condition keeps all 64 bytes inside `buf`.
        let (c, n, q) = unsafe { masks64(buf.as_ptr().add(at)) };
        st.block(buf, at, c, n, q, &mut f);
        at += 64;
    }
    let tail_nl = buf[end - 1] != b'\n';
    if at < end {
        // Zero bytes are not structural, so padding adds no bits.
        let mut tmp = [0u8; 64];
        tmp[..end - at].copy_from_slice(&buf[at..end]);
        // SAFETY: `tmp` holds exactly 64 bytes.
        let (c, mut n, q) = unsafe { masks64(tmp.as_ptr()) };
        if tail_nl {
            // EOF terminates an unterminated final record.
            n |= 1u64 << (end - at);
        }
        st.block(buf, at, c, n, q, &mut f);
    } else if tail_nl {
        let rowq = st.qbase - st.qrow;
        st.finish(buf, end, rowq, &mut f);
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
                 f: &mut impl FnMut(Cell<'b, N>)) {
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
    fn finish<'b>(&mut self, buf: &'b [u8], end: usize, rowq: u32, f: &mut impl FnMut(Cell<'b, N>)) {
        if self.cnt == N - 1 {
            let mut wrapped = 0u32;
            let mut s = self.row_start;
            let mut off = [0u32; 8];
            for i in 0..N {
                let e = if i == N - 1 { end } else { self.ends[i] };
                debug_assert!(s <= e && e <= buf.len() && s >= 1);
                // A quoted field has two or more bytes; `s >= 1` keeps the
                // byte before an empty field's separator in bounds.
                wrapped += 2 * u32::from(e - s >= 2 && buf[s] == b'"' && buf[e - 1] == b'"');
                off[i] = e as u32;
                s = e + 1;
            }
            off[7] = self.row_start as u32;
            if wrapped == rowq {
                // SAFETY: separators were visited in increasing order from
                // the record's first byte (>= 1) to `end <= buf.len()`.
                f(unsafe { Cell::new(buf.as_ptr(), off) });
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
    use super::{cells_window, rows, rows_simple};

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

    /// The cell scanners need a record boundary at or after byte 1: scan the
    /// input behind one LF.
    fn check<const N: usize>(buf: &[u8]) {
        let want = owned::<N>(buf, false);
        assert_eq!(owned::<N>(buf, true), want, "input={buf:?}");
        let mut led = vec![b'\n'];
        led.extend_from_slice(buf);
        let simple = collect::<N>(&led, |b, f| rows_simple::<N>(b, 1, b.len(), |c| f(c.fields())));
        assert_eq!(simple, want, "simple input={buf:?}");
        for window in [64, 128, 192, 256] {
            let got = collect::<N>(&led, |b, f| cells_window::<N>(b, 1, b.len(), window, |c| f(c.fields())));
            assert_eq!(got, want, "window={window} input={buf:?}");
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
    fn open_quote_state_never_leaks_into_a_valid_verdict() {
        // A record leaving a quote open, then records whose quotes would
        // balance it, at every offset modulo 64 and across window ends.
        for tail in [
            b"de\",y\n".as_slice(), b"\"de\",y\n", b"d,\"y\"\n", b"\"\",y\n", b"de\",\"y\n", b"x,y\n",
        ] {
            for shift in 0..130usize {
                let mut input = Vec::new();
                input.extend(std::iter::repeat_n(b'x', shift % 61));
                input.extend_from_slice(b",w\n");
                input.extend(std::iter::repeat_n(b'v', shift));
                input.extend_from_slice(b",\"abc\n");
                input.extend_from_slice(tail);
                input.extend_from_slice(b"\"p\",q\n");
                check::<2>(&input);
            }
        }
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
            let want: Vec<u32> = (0..64).filter(|i| m >> i & 1 == 1).map(|i| 4096 + i).collect();
            for k in [1, 2, 4, 7, 8] {
                let (mut a, mut b) = ([0u16; 72], [0u32; 72]);
                // SAFETY: 72 slots cover max(k, popcount) for any mask.
                unsafe {
                    super::flatten_tz(a.as_mut_ptr(), 4096, m, k);
                    super::flatten_rev(b.as_mut_ptr(), 4096, m, k);
                }
                let a: Vec<u32> = a[..want.len()].iter().map(|&x| x as u32).collect();
                assert_eq!(a, want, "tz m={m:#x} k={k}");
                assert_eq!(&b[..want.len()], &want[..], "rev m={m:#x} k={k}");
            }
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
