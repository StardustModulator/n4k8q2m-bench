//! Raw-user grouping. Each raw-user bin groups its own rows once and names
//! each distinct raw user once. A user named by itself is a person rooted in
//! its own bin: a name's person bin is that name's raw bin, because both use
//! the top bits of the same hash. A renamed user hands only a span (bin,
//! group) to its name's bin. Rows are copied again only for persons that
//! actually span several raw users. Event rows with key 0 are dead (killed by
//! dedup in place) and are dropped here.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::Histories;
use crate::links::Aliases;
use crate::stream::{consume, hash};

#[derive(Clone, Copy)]
struct Span {
    name: u64,
    bin: u32,
    group: u32,
}

struct Table {
    slots: Vec<(u64, u32)>,
    mask: usize,
}

impl Table {
    fn new(n: usize) -> Self {
        let cap = (2 * n).next_power_of_two().max(16);
        Self { slots: vec![(0, 0); cap], mask: cap - 1 }
    }
    // Keys are nonzero. The top hash bits chose the bin; the low bits are free.
    #[inline(always)]
    fn find(&self, key: u64) -> usize {
        let mut i = hash(key) as usize & self.mask;
        loop {
            let k = self.slots[i].0;
            if k == key || k == 0 {
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn get(&self, key: u64) -> Option<usize> {
        let (k, g) = self.slots[self.find(key)];
        (k != 0).then_some(g as usize)
    }
    #[cold]
    fn grow(&mut self) {
        let n = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![(0, 0); n]);
        self.mask = self.slots.len() - 1;
        for (k, g) in old {
            if k != 0 {
                let i = self.find(k);
                self.slots[i] = (k, g);
            }
        }
    }
    #[inline(always)]
    fn intern(&mut self, key: u64, keys: &mut Vec<u64>, ne: &mut Vec<u32>, nc: &mut Vec<u32>) -> usize {
        let mut i = self.find(key);
        if self.slots[i].0 == key {
            return self.slots[i].1 as usize;
        }
        if keys.len() * 4 >= self.slots.len() * 3 {
            self.grow();
            i = self.find(key);
        }
        let g = keys.len();
        self.slots[i] = (key, g as u32);
        keys.push(key);
        ne.push(0);
        nc.push(0);
        g
    }
}

/// One raw-user bin after grouping. Group g's rows are ev[ev_at[g]..ev_at[g+1]]
/// and cl[cl_at[g]..cl_at[g+1]].
struct Bin {
    keys: Vec<u64>,
    renamed: Vec<bool>,
    ev_at: Vec<u32>,
    cl_at: Vec<u32>,
    table: Table,
    ev: Vec<u64>,
    cl: Vec<Hit>,
    // Taken from the owned vectors once grouping is done; afterwards the rows
    // are accessed only through these.
    ev_ptr: *mut u64,
    cl_ptr: *mut Hit,
}

// SAFETY: after grouping, a bin's rows are touched only through ev_ptr/cl_ptr.
// The job that owns bin b sorts the ranges of b's groups that are not
// renamed; other jobs only read the ranges of b's renamed groups. A group is
// either renamed or not, so the ranges are disjoint, and the vectors are never
// resized or dropped while those jobs run.
unsafe impl Send for Bin {}
unsafe impl Sync for Bin {}

impl Bin {
    fn empty() -> Self {
        let mut b = Bin {
            keys: Vec::new(), renamed: Vec::new(), ev_at: vec![0], cl_at: vec![0], table: Table::new(0),
            ev: Vec::new(), cl: Vec::new(), ev_ptr: std::ptr::null_mut(), cl_ptr: std::ptr::null_mut(),
        };
        b.ev_ptr = b.ev.as_mut_ptr();
        b.cl_ptr = b.cl.as_mut_ptr();
        b
    }
    #[inline(always)]
    fn ev_range(&self, g: usize) -> (usize, usize) {
        (self.ev_at[g] as usize, self.ev_at[g + 1] as usize)
    }
    #[inline(always)]
    fn cl_range(&self, g: usize) -> (usize, usize) {
        (self.cl_at[g] as usize, self.cl_at[g + 1] as usize)
    }
    /// SAFETY: see the Sync impl; the caller owns group g's ranges.
    #[inline(always)]
    unsafe fn rows_mut(&self, g: usize) -> (&mut [u64], &mut [Hit]) {
        let (e0, e1) = self.ev_range(g);
        let (c0, c1) = self.cl_range(g);
        unsafe {
            (std::slice::from_raw_parts_mut(self.ev_ptr.add(e0), e1 - e0),
             std::slice::from_raw_parts_mut(self.cl_ptr.add(c0), c1 - c0))
        }
    }
    /// SAFETY: see the Sync impl; group g is renamed (or owned by the caller).
    #[inline(always)]
    unsafe fn rows(&self, g: usize) -> (&[u64], &[Hit]) {
        let (e0, e1) = self.ev_range(g);
        let (c0, c1) = self.cl_range(g);
        unsafe {
            (std::slice::from_raw_parts(self.ev_ptr.add(e0), e1 - e0),
             std::slice::from_raw_parts(self.cl_ptr.add(c0), c1 - c0))
        }
    }
}

/// Group one raw-user bin and name each distinct raw user once.
fn gather(mut parts: Vec<Histories>, aliases: &Aliases, shift: u32, bin: usize, spans: &mut [Vec<Span>]) -> Bin {
    let ne: usize = parts.iter().map(|p| p.ev.len()).sum();
    let nc: usize = parts.iter().map(|p| p.cl.len()).sum();
    if ne + nc == 0 {
        return Bin::empty();
    }
    let mut table = Table::new((ne + nc) / 6 + 16);
    let mut keys = Vec::with_capacity((ne + nc) / 6 + 16);
    let (mut ecount, mut ccount) = (Vec::with_capacity(keys.capacity()), Vec::with_capacity(keys.capacity()));
    // The key field is ours now: it carries the group number to the scatter.
    // A row dedup killed in place has key 0 and is skipped here, not copied.
    let mut live = 0;
    for p in &mut parts {
        for e in &mut p.ev {
            if e.key == 0 {
                e.key = u64::MAX;
                continue;
            }
            let g = table.intern(e.key, &mut keys, &mut ecount, &mut ccount);
            ecount[g] += 1;
            e.key = g as u64;
            live += 1;
        }
        for c in &mut p.cl {
            let g = table.intern(c.key, &mut keys, &mut ecount, &mut ccount);
            ccount[g] += 1;
            c.key = g as u64;
        }
    }
    let groups = keys.len();
    let mut ev_at = Vec::with_capacity(groups + 1);
    let mut cl_at = Vec::with_capacity(groups + 1);
    let (mut es, mut cs) = (0u32, 0u32);
    for g in 0..groups {
        ev_at.push(es);
        cl_at.push(cs);
        // Reuse the counts as scatter cursors.
        let (e, c) = (ecount[g], ccount[g]);
        ecount[g] = es;
        ccount[g] = cs;
        es += e;
        cs += c;
    }
    ev_at.push(es);
    cl_at.push(cs);
    let mut ev = vec![0u64; live];
    let mut cl = vec![Hit { id: 0, ts: 0, camp: 0 }; nc];
    for p in parts {
        for e in p.ev {
            if e.key == u64::MAX {
                continue;
            }
            let at = &mut ecount[e.key as usize];
            ev[*at as usize] = e.val;
            *at += 1;
        }
        for c in p.cl {
            let at = &mut ccount[c.key as usize];
            cl[*at as usize] = c.hit;
            *at += 1;
        }
    }
    let mut renamed = vec![false; groups];
    for g in 0..groups {
        let name = aliases.resolve(keys[g]);
        if name != keys[g] {
            renamed[g] = true;
            spans[(hash(name) >> shift) as usize].push(Span { name, bin: bin as u32, group: g as u32 });
        }
    }
    let mut b = Bin { keys, renamed, ev_at, cl_at, table, ev, cl, ev_ptr: std::ptr::null_mut(), cl_ptr: std::ptr::null_mut() };
    b.ev_ptr = b.ev.as_mut_ptr();
    b.cl_ptr = b.cl.as_mut_ptr();
    b
}

#[derive(Default)]
struct Scratch {
    ev: Vec<u64>,
    cl: Vec<Hit>,
    root: Vec<bool>,
}

/// Sessionize every person whose name lives in bin b.
fn finish(b: usize, bins: &[Bin], mut spans: Vec<Span>, acc: &mut Acc, sc: &mut Scratch,
          cats: &[Vec<u8>], slow: &SlowIntern, first_day: i32) {
    let me = &bins[b];
    let groups = me.keys.len();
    spans.sort_unstable_by_key(|s| s.name);
    // A local group that also receives spans is processed with them.
    sc.root.clear();
    sc.root.resize(groups, false);
    for s in &spans {
        if let Some(g) = me.table.get(s.name) {
            sc.root[g] = true;
        }
    }
    for g in 0..groups {
        if me.renamed[g] || sc.root[g] || me.ev_at[g] == me.ev_at[g + 1] {
            continue;
        }
        // SAFETY: g is neither renamed nor a span target, so only this job
        // touches its ranges.
        let (ev, cl) = unsafe { me.rows_mut(g) };
        acc.person(me.keys[g], ev, cl, cats, slow, first_day);
    }
    let mut i = 0;
    while i < spans.len() {
        let name = spans[i].name;
        let mut j = i + 1;
        while j < spans.len() && spans[j].name == name {
            j += 1;
        }
        sc.ev.clear();
        sc.cl.clear();
        if let Some(g) = me.table.get(name) {
            // SAFETY: the root group is not renamed and belongs to this job.
            let (ev, cl) = unsafe { me.rows(g) };
            sc.ev.extend_from_slice(ev);
            sc.cl.extend_from_slice(cl);
        }
        for s in &spans[i..j] {
            // SAFETY: span groups are renamed, hence only ever read.
            let (ev, cl) = unsafe { bins[s.bin as usize].rows(s.group as usize) };
            sc.ev.extend_from_slice(ev);
            sc.cl.extend_from_slice(cl);
        }
        if !sc.ev.is_empty() {
            acc.person(name, &mut sc.ev, &mut sc.cl, cats, slow, first_day);
        }
        i = j;
    }
}

/// Group raw-user bins (top `64 - shift` hash bits) into persons and sessionize
/// them. Returns one accumulator per worker.
#[allow(clippy::too_many_arguments)]
pub(super) fn run(by_user: Vec<Vec<Histories>>, aliases: &Aliases, threads: usize, shift: u32,
                  acc: impl Fn() -> Acc + Sync, cats: &[Vec<u8>], slow: &SlowIntern, first_day: i32) -> Vec<Acc> {
    let nbins = by_user.len();
    let t0 = std::time::Instant::now();
    let mut jobs: Vec<(usize, Vec<Histories>)> = by_user.into_iter().enumerate().collect();
    jobs.sort_unstable_by_key(|(_, p)| std::cmp::Reverse(p.iter().map(|h| h.ev.len() + h.cl.len()).sum::<usize>()));
    let grouped = consume(jobs, threads, || (Vec::new(), (0..nbins).map(|_| Vec::new()).collect::<Vec<Vec<Span>>>()),
        |(done, spans): &mut (Vec<(usize, Bin)>, Vec<Vec<Span>>), (b, parts)| {
            done.push((b, gather(parts, aliases, shift, b, spans)));
        });
    let mut bins: Vec<Option<Bin>> = (0..nbins).map(|_| None).collect();
    let mut incoming: Vec<Vec<Span>> = (0..nbins).map(|_| Vec::new()).collect();
    for (done, spans) in grouped {
        for (b, bin) in done {
            bins[b] = Some(bin);
        }
        for (dst, mut src) in incoming.iter_mut().zip(spans) {
            dst.append(&mut src);
        }
    }
    let bins: Vec<Bin> = bins.into_iter().map(|b| b.unwrap_or_else(Bin::empty)).collect();
    let t1 = std::time::Instant::now();
    let mut jobs: Vec<(usize, Vec<Span>)> = incoming.into_iter().enumerate().collect();
    jobs.sort_unstable_by_key(|(b, s)| std::cmp::Reverse(bins[*b].ev.len() + s.len()));
    let bins = &bins;
    let out = consume(jobs, threads, || (acc(), Scratch::default()),
        |(a, sc), (b, spans)| finish(b, bins, spans, a, sc, cats, slow, first_day))
        .into_iter().map(|(a, _)| a).collect();
    if std::env::var_os("STAGES").is_some() {
        eprintln!("persons gather={} finish={}", (t1 - t0).as_millis(), t1.elapsed().as_millis());
    }
    out
}
