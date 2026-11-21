//! Bounded-memory Parquet output. Each worker owns a part file and writes one
//! row group per GROUP rows with a single writev from reusable scratch.
//!
//! Columns are dictionary encoded without hashing or sorting. Strings get one
//! dictionary entry per run of equal values (a person's rows are adjacent) and
//! RLE runs of 16-bit indices. Integers get the dense range [origin, origin+K)
//! followed by the out-of-range values in row order, and bit-packed indices of
//! minimal width. Integer width then only sizes the dictionary page, so user
//! counters are always INT64 and the table is read once, with no pre-pass.

use super::{SlowIntern, Tables, UserRow};
use crate::atom::{is_slow, slow_index, write_id};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, IoSlice, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

const GROUP: usize = 16_384; // keeps a row group's rows cache resident
const MIN_PART: usize = 32_768;
const CREATOR: &[u8] = b"shopping-sessions-solve version 2";
const MAGIC: &[u8] = b"PAR1";
const PLAIN: i32 = 0;
const RLE: i32 = 3;
const DICT: i32 = 8; // RLE_DICTIONARY
// Dense dictionary ranges (origin, K): a typical row group then needs 4-bit
// session, 7-bit event and ~11-bit revenue indices.
const SESSIONS: (i64, usize) = (1, 14);
const EVENTS: (i64, usize) = (1, 64);
const REVENUE: (i64, usize) = (0, 1);
const _: () = assert!(GROUP + 256 <= 1 << 16); // indices fit u16 and 16-bit runs

fn invalid(s: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}

fn var(mut x: u64, b: &mut [u8]) -> usize {
    let mut n = 0;
    while x >= 128 {
        b[n] = x as u8 | 128;
        x >>= 7;
        n += 1;
    }
    b[n] = x as u8;
    n + 1
}

fn bitlen(x: usize) -> u32 {
    usize::BITS - x.leading_zeros()
}

#[derive(Default)]
struct Thrift(Vec<u8>);

impl Thrift {
    fn uvar(&mut self, v: u64) {
        let mut b = [0; 10];
        let n = var(v, &mut b);
        self.0.extend_from_slice(&b[..n]);
    }
    fn tag(&mut self, delta: u8, ty: u8) {
        self.0.push(delta << 4 | ty);
    }
    fn stop(&mut self) {
        self.0.push(0);
    }
    fn zz(&mut self, v: i64) {
        self.uvar(((v << 1) ^ (v >> 63)) as u64);
    }
    fn i32(&mut self, d: u8, v: i32) {
        self.tag(d, 5);
        self.zz(i64::from(v));
    }
    fn i64(&mut self, d: u8, v: i64) {
        self.tag(d, 6);
        self.zz(v);
    }
    fn bytes(&mut self, s: &[u8]) {
        self.uvar(s.len() as u64);
        self.0.extend_from_slice(s);
    }
    fn binary(&mut self, d: u8, s: &[u8]) {
        self.tag(d, 8);
        self.bytes(s);
    }
    fn list(&mut self, d: u8, n: usize, ty: u8) {
        self.tag(d, 9);
        if n < 15 {
            self.0.push((n as u8) << 4 | ty);
        } else {
            self.0.push(0xf0 | ty);
            self.uvar(n as u64);
        }
    }
}

#[derive(Clone, Copy)]
enum Kind { I32, I64, Date, Str }

#[derive(Clone, Copy)]
struct Col(&'static str, Kind, bool);

impl Col {
    fn integer(name: &'static str, bits: u64) -> Self {
        // Aggregates are nonnegative. Negative synthetic inputs simply stay i64.
        Self(name, if bits <= i32::MAX as u64 { Kind::I32 } else { Kind::I64 }, false)
    }
    fn physical(self) -> i32 {
        match self.1 { Kind::I32 | Kind::Date => 1, Kind::I64 => 2, Kind::Str => 6 }
    }
    fn schema(self, t: &mut Thrift) {
        t.i32(1, self.physical());
        t.i32(2, i32::from(self.2)); // required = 0, optional = 1
        t.binary(1, self.0.as_bytes());
        let annotation = match self.1 {
            Kind::Date => Some((6, 6)), // converted DATE, logical DATE
            Kind::Str => Some((0, 1)),  // converted UTF8, logical STRING
            _ => None,
        };
        if let Some((converted, logical)) = annotation {
            t.i32(2, converted);
            t.tag(4, 12); // SchemaElement.logicalType
            t.tag(logical, 12);
            t.stop();
            t.stop();
        }
        t.stop();
    }
}

fn page_header(t: &mut Thrift, dict: bool, encoding: i32, n: usize, bytes: usize) -> io::Result<()> {
    let bytes = i32::try_from(bytes).map_err(|_| invalid("Parquet page exceeds INT32 size"))?;
    let n = i32::try_from(n).map_err(|_| invalid("Parquet page exceeds INT32 value count"))?;
    t.i32(1, if dict { 2 } else { 0 });
    t.i32(1, bytes);
    t.i32(1, bytes); // UNCOMPRESSED
    t.tag(if dict { 4 } else { 2 }, 12); // dictionary_page_header=7, data_page_header=5
    t.i32(1, n);
    t.i32(1, encoding);
    if !dict {
        t.i32(1, RLE);
        t.i32(1, RLE);
    }
    t.stop();
    t.stop();
    Ok(())
}

fn writev(file: &mut File, mut pending: &mut [IoSlice<'_>]) -> io::Result<()> {
    loop {
        while pending.first().is_some_and(|b| b.is_empty()) {
            pending = &mut pending[1..];
        }
        if pending.is_empty() { return Ok(()); }
        match file.write_vectored(pending) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => IoSlice::advance_slices(&mut pending, n),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// One column chunk: an optional dictionary page (entry count and PLAIN body)
/// and one data page. A data body is levels, then values; unused parts are empty.
struct Chunk<'a> {
    dict: Option<(usize, [&'a [u8]; 2])>,
    enc: i32,
    body: [&'a [u8]; 4],
}

impl<'a> Chunk<'a> {
    fn plain(values: &'a [u8]) -> Self {
        Self { dict: None, enc: PLAIN, body: [values, &[], &[], &[]] }
    }
}

const MAX_COLS: usize = 7;

struct Writer<'a> {
    f: File,
    cols: &'a [Col],
    pos: i64,
    groups: usize,
    rows: i64,
    meta: Thrift,
    hdr: Thrift,
}

impl<'a> Writer<'a> {
    fn new(f: File, cols: &'a [Col]) -> Self {
        debug_assert!(cols.len() <= MAX_COLS);
        Self { f, cols, pos: 0, groups: 0, rows: 0, meta: Thrift::default(), hdr: Thrift::default() }
    }

    fn group(&mut self, n: usize, chunks: &[Chunk<'_>]) -> io::Result<()> {
        debug_assert_eq!(self.cols.len(), chunks.len());
        let mut h = std::mem::take(&mut self.hdr);
        h.0.clear();
        let mut ends = [(0, 0); MAX_COLS];
        for (c, e) in chunks.iter().zip(&mut ends) {
            if let Some((entries, d)) = c.dict {
                page_header(&mut h, true, PLAIN, entries, d[0].len() + d[1].len())?;
            }
            e.0 = h.0.len();
            page_header(&mut h, false, c.enc, n, c.body.iter().map(|b| b.len()).sum())?;
            e.1 = h.0.len();
        }
        let mut iov = [IoSlice::new(&[]); 1 + MAX_COLS * 8];
        let mut k = 0;
        if self.pos == 0 {
            iov[0] = IoSlice::new(MAGIC);
            k = 1;
            self.pos = MAGIC.len() as i64;
        }
        let t = &mut self.meta;
        t.list(1, chunks.len(), 12);
        let (mut at, mut total) = (0, 0);
        for ((c, &(d, p)), &col) in chunks.iter().zip(&ends).zip(self.cols) {
            let start = self.pos;
            let mut parts = [&[][..]; 8];
            let mut m = 0;
            if let Some((_, dict)) = c.dict {
                parts[..3].copy_from_slice(&[&h.0[at..d], dict[0], dict[1]]);
                m = 3;
            }
            parts[m] = &h.0[d..p];
            parts[m + 1..m + 5].copy_from_slice(&c.body);
            let mut data = start;
            for (i, b) in parts[..m + 5].iter().enumerate() {
                if i == m { data = self.pos; }
                iov[k] = IoSlice::new(b);
                k += 1;
                self.pos += b.len() as i64;
            }
            at = p;
            let size = self.pos - start;
            t.i64(2, 0); // no duplicated ColumnMetaData outside the footer
            t.tag(1, 12);
            t.i32(1, col.physical());
            let enc: &[i32] = if c.dict.is_some() { &[PLAIN, RLE, DICT] }
                else if col.2 { &[PLAIN, RLE] } else { &[PLAIN] };
            t.list(1, enc.len(), 5);
            for &e in enc { t.zz(i64::from(e)); }
            t.list(1, 1, 8);
            t.bytes(col.0.as_bytes()); // exactly the leaf name from the schema
            t.i32(1, 0);
            t.i64(1, n as i64);
            t.i64(1, size);
            t.i64(1, size);
            t.i64(2, data); // ColumnMetaData.data_page_offset = 9
            if c.dict.is_some() { t.i64(2, start); } // dictionary_page_offset = 11
            t.stop();
            t.stop();
            total += size;
        }
        t.i64(1, total);
        t.i64(1, n as i64);
        t.i64(3, total); // RowGroup.total_compressed_size = 6 (5 is file_offset)
        t.stop();
        self.rows += n as i64;
        self.groups += 1;
        writev(&mut self.f, &mut iov[..k])?;
        self.hdr = h;
        Ok(())
    }

    fn finish(mut self) -> io::Result<()> {
        let mut head = Thrift::default();
        head.i32(1, 1);
        head.list(1, 1 + self.cols.len(), 12);
        head.binary(4, b"schema"); // no type or repetition_type on the root
        head.i32(1, self.cols.len() as i32);
        head.stop();
        for &c in self.cols { c.schema(&mut head); }
        head.i64(1, self.rows);
        head.list(1, self.groups, 12); // zero groups, with a real schema, for empty tables
        let mut tail = Thrift::default();
        tail.binary(2, CREATOR);
        tail.stop();
        let len = u32::try_from(head.0.len() + self.meta.0.len() + tail.0.len())
            .map_err(|_| invalid("Parquet footer too large"))?.to_le_bytes();
        let magic = if self.pos == 0 { MAGIC } else { &[] };
        writev(&mut self.f, &mut [magic, &head.0, &self.meta.0, &tail.0, &len, MAGIC].map(IoSlice::new))
    }
}

/// PLAIN BYTE_ARRAY values in a buffer whose length is its high-water mark,
/// so ids can be formatted in place.
#[derive(Default)]
struct Strs { buf: Vec<u8>, len: usize }

impl Strs {
    fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }
    fn room(&mut self, n: usize) -> &mut [u8] {
        if self.buf.len() < self.len + n {
            self.buf.resize((self.len + n).max(2 * self.buf.len()), 0);
        }
        &mut self.buf[self.len..]
    }
    fn push(&mut self, s: &[u8]) {
        // A string too long for i32 makes its page too long; page_header rejects it.
        let room = self.room(4 + s.len());
        room[..4].copy_from_slice(&(s.len() as u32).to_le_bytes());
        room[4..4 + s.len()].copy_from_slice(s);
        self.len += 4 + s.len();
    }
}

/// Bit i of `out[i / 64]` marks a row that starts a run: row 0 and every row
/// that `same` deems different from the row before. Bit n marks the end.
fn run_starts<R>(rows: &[R], same: impl Fn(&R, &R) -> bool, out: &mut Vec<u64>) {
    out.clear();
    out.resize(rows.len() / 64 + 1, 0);
    let mut prev = None;
    for (block, m) in rows.chunks(64).zip(out.iter_mut()) {
        for (j, r) in block.iter().enumerate() {
            *m |= u64::from(prev.is_none_or(|p| !same(p, r))) << j;
            prev = Some(r);
        }
    }
    out[rows.len() / 64] |= 1 << (rows.len() % 64);
}

/// Append the RLE run of `len` copies of the 16-bit index `count` at `at`, a
/// varint of at most three bytes as runs are at most GROUP long. Returns the
/// new end. Needs 8 bytes of room at `at`.
#[inline(always)]
fn put_run(out: &mut [u8], at: usize, len: usize, count: usize) -> usize {
    let x = (len as u64) << 1;
    let (v, k) = if x < 1 << 7 { (x, 1) } else if x < 1 << 14 { (x & 127 | 128 | (x >> 7) << 8, 2) }
        else { (x & 127 | 128 | ((x >> 7) & 127 | 128) << 8 | (x >> 14) << 16, 3) };
    out[at..at + 8].copy_from_slice(&(v | (count as u64) << (8 * k)).to_le_bytes());
    at + k + 2
}

/// RLE runs of 16-bit dictionary indices, one per run marked in `starts`;
/// `entry` receives each run's first row to append its dictionary value.
/// Visiting only the marked starts avoids a per-run mispredicted loop exit:
/// runs average a few rows.
fn index_runs<R>(rows: &[R], starts: &[u64], out: &mut Vec<u8>, mut entry: impl FnMut(&R)) -> usize {
    out.clear();
    out.resize(1 + 5 * rows.len() + 8, 0);
    out[0] = 16; // bit width
    let (mut count, mut start, mut at) = (0, 0, 1);
    for (b, &m) in starts.iter().enumerate() {
        let mut m = if b == 0 { m & !1 } else { m };
        while m != 0 {
            let end = b * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            entry(&rows[start]);
            at = put_run(out, at, end - start, count);
            count += 1;
            start = end;
        }
    }
    out.truncate(at);
    count
}

/// index_runs for user ids, with every position in a local: through the
/// closure the compiler would reload and store them for each run.
fn user_ids(rows: &[UserRow], starts: &[u64], slow: &SlowIntern, dict: &mut Strs, out: &mut Vec<u8>) -> usize {
    let n = rows.len();
    out.clear();
    out.resize(1 + 5 * n + 8, 0);
    out[0] = 16; // bit width
    // A packed id takes at most 13 bytes, and write_id wants 24 writable.
    dict.len = 0;
    dict.room(13 * n + 28);
    let (mut count, mut start, mut at, mut dp) = (0, 0, 1, 0);
    for (b, &m) in starts.iter().enumerate() {
        let mut m = if b == 0 { m & !1 } else { m };
        while m != 0 {
            let end = b * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let key = rows[start].key;
            if is_slow(key) {
                dict.len = dp;
                dict.push(&slow.strings[slow_index(key)]);
                dict.room(13 * n + 28);
                dp = dict.len;
            } else {
                let buf = &mut dict.buf[dp..dp + 28];
                let len = write_id(key, (&mut buf[4..]).try_into().unwrap());
                buf[..4].copy_from_slice(&(len as u32).to_le_bytes());
                dp += 4 + len;
            }
            at = put_run(out, at, end - start, count);
            count += 1;
            start = end;
        }
    }
    dict.len = dp;
    out.truncate(at);
    count
}

/// Bit-pack 16-bit indices, 8 values into W bytes, LSB first.
fn pack_w<const W: usize>(idx: &[u16], out: &mut Vec<u8>) {
    debug_assert!(idx.len() % 8 == 0 && idx.iter().all(|&x| usize::from(x) >> W == 0));
    // Squeeze four 16-bit lanes of a word into 4W low bits in two steps.
    const L: u64 = 0x0000_ffff_0000_ffff;
    let squeeze = |x: u64| {
        let x = (x & L) | ((x >> 16) & L) << W;
        (x & 0xffff_ffff) | (x >> 32) << (2 * W)
    };
    let len = idx.len() / 8 * W;
    out.clear();
    out.reserve(len + 16);
    let dst = out.as_mut_ptr();
    for (g, v) in idx.chunks_exact(8).enumerate() {
        let lo = squeeze(u64::from(v[0]) | u64::from(v[1]) << 16 | u64::from(v[2]) << 32 | u64::from(v[3]) << 48);
        let hi = squeeze(u64::from(v[4]) | u64::from(v[5]) << 16 | u64::from(v[6]) << 32 | u64::from(v[7]) << 48);
        let acc = u128::from(lo) | u128::from(hi) << (4 * W);
        // SAFETY: g * W + 16 <= len - W + 16 <= capacity; later groups
        // overwrite the zero tail of this store.
        unsafe { dst.add(g * W).cast::<u128>().write_unaligned(acc.to_le()) };
    }
    // SAFETY: the loop initialized out[..len].
    unsafe { out.set_len(len) };
}

fn pack(idx: &[u16], w: u32, out: &mut Vec<u8>) {
    macro_rules! by_width {
        ($($w:literal)*) => { match w { $($w => pack_w::<$w>(idx, out),)* _ => unreachable!() } };
    }
    by_width!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
}

/// A bit-packed run of dictionary indices, preceded by its bit width.
#[derive(Default)]
struct Packed { bits: Vec<u8>, head: [u8; 12], hlen: usize }

impl Packed {
    /// `idx` holds a multiple of 8 indices, none above `max`.
    fn finish(&mut self, idx: &[u16], max: usize) {
        let w = bitlen(max).max(1);
        pack(idx, w, &mut self.bits);
        self.head[0] = w as u8;
        self.hlen = 1 + var(((idx.len() / 8) as u64) << 1 | 1, &mut self.head[1..]);
    }
    fn head(&self) -> &[u8] {
        &self.head[..self.hlen]
    }
}

/// Dictionary: `dense` (K values from origin), then the other values in row order.
#[derive(Default)]
struct Ints { p: Packed, exc: Vec<i64>, narrow: Vec<u8>, entries: usize }

impl Ints {
    fn encode<R>(&mut self, idx: &mut Vec<u16>, rows: &[R], (origin, k): (i64, usize), wide: bool,
        get: impl Fn(&R) -> i64) {
        self.exc.clear();
        idx.clear();
        idx.extend(rows.iter().map(|r| {
            let v = get(r);
            let d = v.wrapping_sub(origin) as u64;
            if d < k as u64 { d as u16 } else { spill(&mut self.exc, k, d, v) }
        }));
        idx.resize(rows.len().next_multiple_of(8), 0);
        self.seal(idx, k, wide);
    }
    fn seal(&mut self, idx: &[u16], k: usize, wide: bool) {
        self.entries = k + self.exc.len();
        self.p.finish(idx, self.entries - 1);
        if !wide || cfg!(target_endian = "big") {
            self.narrow.clear();
            for &v in &self.exc {
                if wide { self.narrow.extend_from_slice(&v.to_le_bytes()); }
                else { self.narrow.extend_from_slice(&(v as i32).to_le_bytes()); }
            }
        }
    }
    fn chunk<'a>(&'a self, dense: &'a [u8], wide: bool) -> Chunk<'a> {
        let exc = if wide && cfg!(target_endian = "little") {
            // SAFETY: i64 has no padding; on little-endian targets its bytes
            // are exactly PLAIN INT64.
            unsafe { std::slice::from_raw_parts(self.exc.as_ptr().cast::<u8>(), self.exc.len() * 8) }
        } else { &self.narrow };
        Chunk { dict: Some((self.entries, [dense, exc])), enc: DICT, body: [self.p.head(), &self.p.bits, &[], &[]] }
    }
}

fn dense(origin: i64, k: usize, wide: bool) -> Vec<u8> {
    let mut d = Vec::new();
    for v in (0..k as i64).map(|i| origin + i) {
        if wide { d.extend_from_slice(&v.to_le_bytes()); } else { d.extend_from_slice(&(v as i32).to_le_bytes()); }
    }
    d
}

/// Optional top_category: bitmap definition levels plus dictionary indices.
#[derive(Default)]
struct Top { p: Packed, bitmap: Vec<u8>, levels: [u8; 16], llen: usize, defined: usize }

impl Top {
    /// `idx` holds the `e` defined values, padded to a multiple of 8.
    fn seal(&mut self, idx: &[u16], e: usize, n: usize, cats: usize) -> io::Result<()> {
        if idx[..e].iter().any(|&t| usize::from(t) >= cats) { return Err(invalid("invalid top_category index")); }
        self.defined = e;
        let end = if e == 0 || e == n {
            let end = 4 + var((n as u64) << 1, &mut self.levels[4..]);
            self.levels[end] = u8::from(e != 0);
            self.bitmap.clear();
            end + 1
        } else {
            4 + var(self.bitmap.len() as u64 * 2 + 1, &mut self.levels[4..])
        };
        self.levels[..4].copy_from_slice(&((end - 4 + self.bitmap.len()) as u32).to_le_bytes());
        self.llen = end;
        if e != 0 { self.p.finish(idx, cats - 1); }
        Ok(())
    }
    fn chunk<'a>(&'a self, dict: &'a [u8], cats: usize) -> Chunk<'a> {
        let levels = &self.levels[..self.llen];
        if self.defined == 0 {
            return Chunk { dict: None, enc: PLAIN, body: [levels, &[], &[], &[]] };
        }
        Chunk { dict: Some((cats, [dict, &[]])), enc: DICT, body: [levels, &self.bitmap, self.p.head(), &self.p.bits] }
    }
}

fn spill(exc: &mut Vec<i64>, k: usize, d: u64, v: i64) -> u16 {
    if d < k as u64 { return d as u16; }
    exc.push(v);
    (k + exc.len() - 1) as u16
}

/// Per-thread scratch of user row groups. The index columns of day,
/// sessions, events, revenue and top_category sit GROUP apart in one buffer,
/// so the fill loop reaches all from one register: x86 has too few for five
/// pointers next to the loop state.
struct Cols { idx: Vec<u16>, ints: [Ints; 4], top: Top, starts: Vec<u64>, odd: Vec<u64> }

impl Default for Cols {
    fn default() -> Self {
        Self { idx: vec![0; 5 * GROUP], ints: Default::default(), top: Top::default(), starts: Vec::new(),
            odd: Vec::new() }
    }
}

/// Dictionary indices of all integer columns and top_category and the run
/// starts of user ids, in one pass over the rows. Revenue and top_category are
/// branch free; day, sessions and events are almost always in range, and rows
/// where one is not are only marked, then fixed in order after the loop: a
/// call inside it would cost registers, which x86 lacks here.
fn fill(rows: &[UserRow], d: &UserDicts, s: &mut Cols) -> io::Result<()> {
    let n = rows.len();
    assert!(n <= GROUP); // also proves every index below in bounds
    let idx = &mut s.idx[..5 * GROUP];
    let [day, ses, evs, rev] = &mut s.ints;
    for x in [&mut *day, &mut *ses, &mut *evs, &mut *rev] { x.exc.clear(); }
    rev.exc.reserve(n);
    let x3 = rev.exc.spare_capacity_mut();
    let top = &mut s.top;
    top.bitmap.clear();
    top.bitmap.resize(n.div_ceil(64) * 8, 0);
    s.starts.clear();
    s.starts.resize(n / 64 + 1, 0);
    s.odd.clear();
    s.odd.resize(n.div_ceil(64), 0);
    let ((o0, k0), (o1, k1), (o2, k2), (o3, k3)) = (d.day, SESSIONS, EVENTS, REVENUE);
    let (mut er, mut et) = (0, 0);
    let mut prev = rows.first().map_or(0, |r| !r.key);
    let blocks = s.starts.iter_mut().zip(top.bitmap.chunks_exact_mut(8)).zip(s.odd.iter_mut());
    for (b, ((m, defs), odd)) in blocks.enumerate() {
        let (mut bits, mut def, mut out) = (0, 0, 0);
        for i in b * 64..(b * 64 + 64).min(n) {
            let r = &rows[i];
            bits |= u64::from(r.key != prev) << (i % 64);
            prev = r.key;
            let a = i64::from(r.day).wrapping_sub(o0) as u64;
            let b = r.sessions.wrapping_sub(o1) as u64;
            let c = r.events.wrapping_sub(o2) as u64;
            idx[i] = a as u16;
            idx[GROUP + i] = b as u16;
            idx[2 * GROUP + i] = c as u16;
            out |= u64::from((a >= k0 as u64) | (b >= k1 as u64) | (c >= k2 as u64)) << (i % 64);
            let v = r.revenue;
            let x = v.wrapping_sub(o3) as u64;
            let big = x >= k3 as u64;
            // SAFETY: er and et count earlier rows only, so both are below
            // n, which bounds the spare capacity and GROUP.
            unsafe { x3.get_unchecked_mut(er) }.write(v);
            idx[3 * GROUP + i] = if big { (k3 + er) as u16 } else { x as u16 };
            er += usize::from(big);
            let t = r.top;
            let some = t != u16::MAX;
            def |= u64::from(some) << (i % 64);
            *unsafe { idx.get_unchecked_mut(4 * GROUP + et) } = t;
            et += usize::from(some);
        }
        *m = bits;
        defs.copy_from_slice(&def.to_le_bytes());
        *odd = out;
    }
    // SAFETY: the loop above initialized er revenue exceptions.
    unsafe { rev.exc.set_len(er) };
    for (b, &m) in s.odd.iter().enumerate() {
        let mut m = m;
        while m != 0 {
            let i = b * 64 + m.trailing_zeros() as usize;
            m &= m - 1;
            let r = &rows[i];
            let (v0, v1, v2) = (i64::from(r.day), r.sessions, r.events);
            idx[i] = spill(&mut day.exc, k0, v0.wrapping_sub(o0) as u64, v0);
            idx[GROUP + i] = spill(&mut ses.exc, k1, v1.wrapping_sub(o1) as u64, v1);
            idx[2 * GROUP + i] = spill(&mut evs.exc, k2, v2.wrapping_sub(o2) as u64, v2);
        }
    }
    s.starts[n / 64] |= 1 << (n % 64);
    top.bitmap.truncate(n.div_ceil(8));
    let n8 = n.next_multiple_of(8);
    for c in 0..4 { idx[c * GROUP + n..c * GROUP + n8].fill(0); }
    idx[4 * GROUP + et..4 * GROUP + et.next_multiple_of(8)].fill(0);
    let col = |c: usize, len: usize| c * GROUP..c * GROUP + len;
    day.seal(&idx[col(0, n8)], k0, false);
    ses.seal(&idx[col(1, n8)], k1, true);
    evs.seal(&idx[col(2, n8)], k2, true);
    rev.seal(&idx[col(3, n8)], k3, true);
    top.seal(&idx[col(4, et.next_multiple_of(8))], et, n, d.cats_n)
}

/// Per-table constants shared by all user part files.
struct UserDicts { day: (i64, usize), dense: [Vec<u8>; 4], cats: Vec<u8>, cats_n: usize }

impl UserDicts {
    fn new(t: &Tables) -> Self {
        // Every user day has a credited session, so campaign days bound user
        // days. Anything outside still encodes, as a dictionary exception.
        let (lo, hi) = t.camps_out.iter().fold((i32::MAX, i32::MIN), |(l, h), r| (l.min(r.day), h.max(r.day)));
        let day = if lo > hi { (t.users.first().map_or(0, |r| i64::from(r.day)), 16) }
            else { (i64::from(lo), (hi.abs_diff(lo) as usize + 1).min(256)) };
        let mut cats = Strs::default();
        for c in &t.cat_names { cats.push(c); }
        cats.buf.truncate(cats.len);
        Self { day, cats: cats.buf, cats_n: t.cat_names.len(), dense: [dense(day.0, day.1, false), dense(SESSIONS.0, SESSIONS.1, true),
            dense(EVENTS.0, EVENTS.1, true), dense(REVENUE.0, REVENUE.1, true)] }
    }
}

const USER_COLS: [Col; 6] = [Col("user_id", Kind::Str, false), Col("day", Kind::Date, false),
    Col("sessions", Kind::I64, false), Col("events", Kind::I64, false),
    Col("revenue_cents", Kind::I64, false), Col("top_category", Kind::Str, true)];

fn users(file: File, d: &UserDicts, t: &Tables, rows: &[UserRow]) -> io::Result<()> {
    let mut w = Writer::new(file, &USER_COLS);
    let (mut ids, mut runs, mut s) = (Strs::default(), Vec::new(), Cols::default());
    let [dd, ds, de, dr] = &d.dense;
    for r in rows.chunks(GROUP) {
        fill(r, d, &mut s)?;
        let entries = user_ids(r, &s.starts, &t.slow, &mut ids, &mut runs);
        let [day, ses, evs, rev] = &s.ints;
        w.group(r.len(), &[
            Chunk { dict: Some((entries, [ids.bytes(), &[]])), enc: DICT, body: [&runs, &[], &[], &[]] },
            day.chunk(dd, false), ses.chunk(ds, true), evs.chunk(de, true), rev.chunk(dr, true),
            s.top.chunk(&d.cats, d.cats_n),
        ])?;
    }
    w.finish()
}

fn campaign(f: File, t: &Tables) -> io::Result<()> {
    let rows = &t.camps_out;
    let mut b = [0u64; 4];
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for r in rows {
        if usize::from(r.camp) >= t.camps.len() { return Err(invalid("invalid campaign index")); }
        b[0] |= r.sessions as u64; b[1] |= r.purchasing as u64;
        b[2] |= r.revenue as u64; b[3] |= r.assisted as u64;
        lo = lo.min(r.day);
        hi = hi.max(r.day);
    }
    let cols = [Col("campaign_id", Kind::Str, false), Col("channel", Kind::Str, false),
        Col("day", Kind::Date, false), Col::integer("sessions", b[0]),
        Col::integer("purchasing_sessions", b[1]), Col::integer("revenue_cents", b[2]),
        Col::integer("assisted_sessions", b[3])];
    let range = (i64::from(lo), if lo > hi { 1 } else { (hi.abs_diff(lo) as usize + 1).min(256) });
    let dense_day = dense(range.0, range.1, false);
    let mut w = Writer::new(f, &cols);
    let (mut names, mut chans, mut runs, mut starts) = (Strs::default(), Strs::default(), Vec::new(), Vec::new());
    let (mut day, mut idx) = (Ints::default(), Vec::new());
    let mut nums: [Vec<u8>; 4] = Default::default();
    for r in rows.chunks(GROUP) {
        names.len = 0;
        chans.len = 0;
        run_starts(r, |a, b| a.camp == b.camp, &mut starts);
        let entries = index_runs(r, &starts, &mut runs, |row| {
            let c = &t.camps[usize::from(row.camp)];
            names.push(&c.name);
            chans.push(&c.channel);
        });
        day.encode(&mut idx, r, range, false, |r| i64::from(r.day));
        for (i, out) in nums.iter_mut().enumerate() {
            out.clear();
            let wide = matches!(cols[3 + i].1, Kind::I64);
            for row in r {
                let v = [row.sessions, row.purchasing, row.revenue, row.assisted][i];
                if wide { out.extend_from_slice(&v.to_le_bytes()); }
                else { out.extend_from_slice(&(v as i32).to_le_bytes()); }
            }
        }
        w.group(r.len(), &[
            Chunk { dict: Some((entries, [names.bytes(), &[]])), enc: DICT, body: [&runs, &[], &[], &[]] },
            Chunk { dict: Some((entries, [chans.bytes(), &[]])), enc: DICT, body: [&runs, &[], &[], &[]] },
            day.chunk(&dense_day, false),
            Chunk::plain(&nums[0]), Chunk::plain(&nums[1]), Chunk::plain(&nums[2]), Chunk::plain(&nums[3]),
        ])?;
    }
    w.finish()
}

// Never publish an incomplete part. Encode into a hidden file directly under
// the output root, which exists before the table directories do, then move it
// to the name `path` yields. A killed writer can only leave a hidden temp
// file, which the challenge ignores and a subsequent run won't overwrite.
fn staged(root: &Path, tag: &str, path: impl FnOnce() -> io::Result<PathBuf>,
    encode: impl FnOnce(File) -> io::Result<()>) -> io::Result<()> {
    let hidden = |dir: &Path| {
        let mut generation = 0u64;
        loop {
            let tmp = dir.join(format!(".{tag}.{}.{generation}.tmp", std::process::id()));
            match File::create_new(&tmp) {
                Ok(f) => return Ok((tmp, f)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => generation += 1,
                Err(e) => return Err(e),
            }
        }
    };
    let (tmp, file) = hidden(root)?;
    let result = encode(file).and_then(|_| path()).and_then(|path| {
        fs::rename(&tmp, &path).or_else(|_| {
            // A table directory on another filesystem: copy beside it first.
            let (near, _) = hidden(path.parent().unwrap())?;
            let moved = fs::copy(&tmp, &near).and_then(|_| fs::rename(&near, &path));
            if moved.is_err() { let _ = fs::remove_file(&near); }
            moved.and_then(|_| fs::remove_file(&tmp))
        })
    });
    if result.is_err() { let _ = fs::remove_file(tmp); }
    result
}

// Both our footer and the previous single-file writer end with created_by
// followed by STOP. Do not claim a foreign file just because its name matches.
fn ours(path: &Path, legacy: bool) -> io::Result<bool> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let n = size.min(64) as usize;
    if n < 12 { return Ok(false); }
    let mut b = [0; 64];
    f.seek(SeekFrom::End(-(n as i64)))?;
    f.read_exact(&mut b[..n])?;
    if &b[n - 4..n] != MAGIC { return Ok(false); }
    let footer = u32::from_le_bytes(b[n-8..n-4].try_into().unwrap());
    if u64::from(footer) + 12 > size { return Ok(false); }
    let tail = &b[..n - 8];
    let tagged = |name: &[u8]| {
        let l = name.len();
        l + 3 <= tail.len() && l + 3 <= footer as usize && tail[tail.len()-1] == 0
            && tail[tail.len()-l-3..tail.len()-l-1] == [0x28, l as u8]
            && &tail[tail.len()-l-1..tail.len()-1] == name
    };
    Ok(tagged(CREATOR) || (legacy && tagged(b"solve")))
}

fn parts(dir: &Path, count: usize) -> io::Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    fs::create_dir_all(dir)?;
    let mut foreign = HashSet::new();
    let mut prior = Vec::new();
    for item in fs::read_dir(dir)? {
        let item = item?;
        let name = item.file_name();
        let Some(digits) = name.to_str().and_then(|s| s.strip_prefix("part-"))
            .and_then(|s| s.strip_suffix(".parquet")) else { continue };
        let Ok(idx) = digits.parse::<usize>() else { continue };
        if digits != format!("{idx:05}") { continue; }
        if item.file_type()?.is_file() && ours(&item.path(), idx == 0)? {
            prior.push((idx, item.path()));
        } else { foreign.insert(idx); }
    }
    let mut idx = 0;
    let mut paths = Vec::with_capacity(count);
    while paths.len() < count {
        if !foreign.contains(&idx) { paths.push(dir.join(format!("part-{idx:05}.parquet"))); }
        idx += 1;
    }
    Ok((paths, prior.into_iter().filter_map(|(i, p)| (i >= idx).then_some(p)).collect()))
}

pub(super) fn write(dir: &Path, t: &Tables, threads: usize) -> io::Result<()> {
    let n = t.users.len();
    let p = threads.max(1).min(n.div_ceil(MIN_PART).max(1));
    let at = |i: usize| (n / p) * i + i.min(n % p);
    let d = &UserDicts::new(t);
    fs::create_dir_all(dir)?;
    // Workers start encoding at once; making and scanning the table
    // directories (a mkdir alone can take hundreds of µs) overlaps them, and
    // each worker learns its part's name when it is done.
    let stale = thread::scope(|scope| -> io::Result<Vec<PathBuf>> {
        let (mut jobs, mut names) = (Vec::with_capacity(p), Vec::with_capacity(p));
        for i in 0..p {
            let (tx, rx) = mpsc::channel();
            let rows = &t.users[at(i)..at(i + 1)];
            let name = move || rx.recv().map_err(|_| io::Error::other("no name for a user_daily part"));
            let tag = format!("user_daily-{i:05}");
            jobs.push(scope.spawn(move || staged(dir, &tag, name, |f| users(f, d, t, rows))));
            names.push(tx);
        }
        let user = parts(&dir.join("user_daily"), p).map(|(paths, stale)| {
            for (tx, path) in names.iter().zip(paths) { let _ = tx.send(path); }
            stale
        });
        drop(names); // without names, workers discard their files
        let camp = parts(&dir.join("campaign_daily"), 1)
            .and_then(|(mut path, stale)| staged(dir, "campaign_daily", || Ok(path.remove(0)), |f| campaign(f, t))
                .map(|_| stale));
        let done = jobs.into_iter().try_fold((), |_, job| job.join().unwrap());
        let mut stale = user?;
        stale.extend(camp?);
        done.map(|_| stale)
    })?;
    for file in stale { fs::remove_file(file)?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_matches_reference() {
        let mut rand = 0x2545_f491_4f6c_dd1du64;
        for w in 1..=16u32 {
            for n in [8, 16, 64, 8200] {
                let idx: Vec<u16> = (0..n).map(|_| {
                    rand ^= rand << 13; rand ^= rand >> 7; rand ^= rand << 17;
                    (rand >> 20) as u16 & ((1u32 << w) - 1) as u16
                }).collect();
                let mut want = vec![0u8; n * w as usize / 8];
                for (i, &x) in idx.iter().enumerate() {
                    for b in 0..w as usize {
                        let bit = i * w as usize + b;
                        want[bit / 8] |= (((x >> b) & 1) as u8) << (bit % 8);
                    }
                }
                let mut got = Vec::new();
                pack(&idx, w, &mut got);
                assert_eq!(got, want, "w={w} n={n}");
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    // BENCH_DATA=/tmp/ss-20m BENCH_THREADS=8 cargo test --release --bin solve bench_write -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_write() {
        let data = std::env::var("BENCH_DATA").unwrap_or_else(|_| "/tmp/ss-20m".into());
        let threads: usize = std::env::var("BENCH_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
        let t = crate::stream::compute(Path::new(&data), 8).unwrap();
        let root = std::env::temp_dir().join(format!("bench_write.{}", std::process::id()));
        let mut ms = Vec::new();
        for i in 0..reps {
            let dir = root.join(format!("{i}"));
            let s = Instant::now();
            write(&dir, &t, threads).unwrap();
            ms.push(s.elapsed().as_secs_f64() * 1e3);
        }
        let _ = fs::remove_dir_all(&root);
        let d = UserDicts::new(&t);
        let mut enc = f64::MAX;
        for _ in 0..reps {
            let s = Instant::now();
            users(File::create("/dev/null").unwrap(), &d, &t, &t.users).unwrap();
            enc = enc.min(s.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("encode to /dev/null, 1 thread: {enc:.2} ms = {:.2} ns/row", enc * 1e6 / t.users.len() as f64);
        let n = t.users.len();
        let mut par = f64::MAX;
        for _ in 0..reps {
            let s = Instant::now();
            thread::scope(|sc| {
                let hs: Vec<_> = (0..threads).map(|i| { let (d, t) = (&d, &t); sc.spawn(move || {
                    let r = &t.users[n * i / threads..n * (i + 1) / threads];
                    users(File::create("/dev/null").unwrap(), d, t, r).unwrap(); }) }).collect();
                for h in hs { h.join().unwrap(); }
            });
            par = par.min(s.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("encode to /dev/null, {threads} threads: {par:.2} ms");
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        eprintln!("rows={} threads={threads} min={:.2} med={:.2} all={:?}", t.users.len(), sorted[0],
            sorted[sorted.len() / 2], ms.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
        std::mem::forget(t);
    }
}

#[cfg(test)]
mod bench_cols {
    use super::*;
    use std::time::Instant;

    // cargo test --release --bin solve bench_cols -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_cols() {
        let t = crate::stream::compute(Path::new("/tmp/ss-20m"), 8).unwrap();
        let d = UserDicts::new(&t);
        let rows = &t.users[..];
        let time = |name: &str, f: &mut dyn FnMut()| {
            let mut best = f64::MAX;
            for _ in 0..7 { let s = Instant::now(); f(); best = best.min(s.elapsed().as_secs_f64()); }
            eprintln!("{name:>14}: {:6.3} ms {:5.2} ns/row", best * 1e3, best * 1e9 / rows.len() as f64);
        };
        let (mut ints, mut idx, mut out) = (Ints::default(), Vec::new(), Vec::new());
        let mut w = 0;
        time("pack revenue", &mut || for r in rows.chunks(GROUP) {
            ints.encode(&mut idx, r, REVENUE, true, |r| r.revenue);
            w = ints.p.head[0];
            pack(&idx, u32::from(w), &mut out);
        });
        let mut s = Cols::default();
        time("fill", &mut || for r in rows.chunks(GROUP) { fill(r, &d, &mut s).unwrap(); std::hint::black_box(&s.top.p.bits); });
        let (mut cold, mut warm) = (f64::MAX, f64::MAX);
        for _ in 0..5 {
            let (mut a, mut b) = (0.0, 0.0);
            for r in rows.chunks(GROUP) {
                let x = Instant::now(); fill(r, &d, &mut s).unwrap(); a += x.elapsed().as_secs_f64();
                let x = Instant::now(); fill(r, &d, &mut s).unwrap(); b += x.elapsed().as_secs_f64();
            }
            cold = cold.min(a);
            warm = warm.min(b);
        }
        eprintln!("fill cold {:.3} ms warm {:.3} ms", cold * 1e3, warm * 1e3);
        time("campaign", &mut || campaign(File::create("/dev/null").unwrap(), &t).unwrap());
        let (mut ids, mut runs, mut st) = (Strs::default(), Vec::new(), Vec::new());
        time("ids", &mut || for r in rows.chunks(GROUP) {
            run_starts(r, |a, b| a.key == b.key, &mut st);
            user_ids(r, &st, &t.slow, &mut ids, &mut runs);
            std::hint::black_box(&runs);
        });
        std::mem::forget(t);
    }
}
