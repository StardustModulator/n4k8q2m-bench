//! Each event row is written once, straight into its raw user's bin. Dedup
//! sees only a 16-byte candidate per row, partitioned by event ID, and kills
//! losing copies in place. Renaming then compacts each user bin and scatters
//! only renamed rows; neither an all-user interner nor a global sort is needed.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::{histories, Click, Event, Histories};
use crate::clock::Clock;
use crate::links::Aliases;

#[inline(always)]
pub(super) fn hash(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// One per event row with an ID and a parsable ingest, valid or not: dedup
/// precedes validation. `loc` is `row << user_bits | user_bin` of the row's
/// event in its worker's bins, or DEAD when the row failed validation.
#[derive(Clone, Copy)]
struct Cand {
    id: u64,
    sec: u32, // ingest seconds; F1 has no fraction
    loc: u32,
}
const DEAD: u32 = u32::MAX;

/// The exact form of a candidate whose ingest or location does not fit.
#[derive(Clone, Copy)]
struct Wide {
    id: u64,
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
        let part = (hash(id) >> (64 - b.id)) as usize;
        let loc = match at_row {
            None => DEAD as u64,
            Some((bin, row)) => (row as u64) << b.user | bin as u64,
        };
        if (loc < DEAD as u64 || at_row.is_none()) && (0..=u32::MAX as i64).contains(&sec) {
            self.cand[part].push(Cand { id, sec: sec as u32, loc: loc as u32 });
        } else {
            self.push_wide(part, Wide { id, at: sec * 1000, at_row });
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

#[inline(always)]
fn payload(f: &[&[u8]; 7], cx: &Ctx<'_>, clock: &Clock, at: i64, origin: i64) -> Option<u64> {
    // Whole-word compares: a view/cart split costs no branch.
    let t = f[4];
    let purchase = match *t {
        [a, b, c, d] => {
            let w = u32::from_le_bytes([a, b, c, d]);
            if w != VIEW && w != CART { return None; }
            false
        }
        [a, b, c, d, e, g, h, i] if u64::from_le_bytes([a, b, c, d, e, g, h, i]) == PURCHASE => true,
        _ => return None,
    };
    if f[5].is_empty() { return None; }
    let ts = clock.parse(f[2])?;
    if ts < cx.window.start || ts >= cx.window.end || at > ts + DAY_MS {
        return None;
    }
    let pay = if purchase { parse_price(f[6])? | u32::from(product_cat(cx, f[5])) << 24 } else { 0 };
    Some(((ts - origin) as u64) << 32 | u64::from(pay))
}

/// Rows per decode batch. The scanner invokes its callback from several
/// sites; a callback that only stores fields stays small enough to inline,
/// and the decoder below runs as one loop with its constants in registers.
const BATCH: usize = 32;

struct Events<'a, 'c> {
    cx: &'a Ctx<'c>,
    clock: &'a Clock,
    s: &'a Shared,
    origin: i64,
    b: Bits,
}

impl Events<'_, '_> {
    #[inline(never)]
    fn decode(&self, rows: &[[&[u8]; 7]], out: &mut Scan) {
        for f in rows {
            self.row(f, out);
        }
    }

    #[inline(always)]
    fn row(&self, f: &[&[u8]; 7], out: &mut Scan) {
        let Some(id) = id_key(self.s, f[0]) else { return };
        let Some(sec) = self.clock.ingest_s(f[3]) else { return };
        let mut at_row = None;
        if !f[1].is_empty() {
            if let Some(val) = payload(f, self.cx, self.clock, sec * 1000, self.origin) {
                let key = user_key(self.s, f[1]);
                let bin = (hash(key) >> (64 - self.b.user)) as usize;
                let v = &mut out.ev[bin];
                at_row = Some((bin, v.len()));
                v.push(Event { key, val });
            }
        }
        out.candidate(id, sec, at_row, self.b);
    }
}

#[inline]
fn scan_events(buf: &[u8], ev: &Events<'_, '_>, out: &mut Scan) {
    let mut batch = [[&buf[..0]; 7]; BATCH];
    let mut n = 0;
    wire::rows::<7>(buf, |f| {
        batch[n % BATCH] = f;
        n += 1;
        if n == BATCH {
            ev.decode(&batch, out);
            n = 0;
        }
    });
    ev.decode(&batch[..n], out);
}

// All copies of a click are identical. Window membership is existential per
// campaign and credit is a min/max over keys. Both are idempotent: duplicating
// an identical tuple cannot change either result, so clicks skip dedup.
#[inline]
fn scan_clicks(buf: &[u8], cx: &Ctx<'_>, clock: &Clock, s: &Shared, out: &mut Scan, origin: i64, b: Bits) {
    wire::rows::<4>(buf, |f| {
        if f[0].is_empty() || f[1].is_empty() { return; }
        let Some(ts) = clock.parse(f[3]) else { return };
        if ts < cx.window.click_start || ts >= cx.window.end { return; }
        let Some(camp) = campaign_of(cx, f[2]) else { return };
        let key = user_key(s, f[1]);
        let hit = Hit { ts: (ts - origin) as u32, id: user_key(s, f[0]), camp };
        out.cl[(hash(key) >> (64 - b.user)) as usize].push(Click { key, hit });
    });
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
    // exact table still compares full ids. The partition fixed the top bits.
    let words = n.div_ceil(4).next_power_of_two();
    let wbits = words.trailing_zeros();
    let probe = |id: u64| {
        let g = hash(id) << id_bits;
        ((g >> (63 - wbits) >> 1) as usize, 1u64 << ((g >> (58 - wbits)) & 63))
    };
    let mut present = vec![0u64; words];
    let mut repeated = vec![0u64; words];
    for c in parts.iter().flatten() {
        let (i, bit) = probe(c.id);
        let seen = present[i];
        repeated[i] |= seen & bit;
        present[i] = seen | bit;
    }
    let mut suspects = n;
    for i in 0..words {
        suspects -= (present[i] ^ repeated[i]).count_ones() as usize;
    }
    if suspects == 0 { return; }
    // Slot value: part index + 1 in the high half, row in the low half.
    let mask = (suspects * 2).next_power_of_two() - 1;
    let mut table = vec![0u64; mask + 1];
    for (p, part) in parts.iter().enumerate() {
        for (i, c) in part.iter().enumerate() {
            let (w, bit) = probe(c.id);
            if repeated[w] & bit == 0 { continue; }
            let me = ((p as u64 + 1) << 32) | i as u64;
            let mut s = hash(c.id.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize & mask;
            loop {
                let ptr = table[s];
                if ptr == 0 {
                    table[s] = me;
                    break;
                }
                let (q, j) = ((ptr >> 32) as usize - 1, ptr as u32 as usize);
                let old = parts[q][j];
                if old.id == c.id {
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
        (w, c.id, i64::from(c.sec) * 1000, at_row)
    }));
    let full = wide.iter().enumerate().flat_map(|(w, v)| v.iter().map(move |x| (w, x.id, x.at, x.at_row)));
    for (w, id, at, at_row) in compact.chain(full) {
        let kill = |w: usize, at_row: Option<(usize, usize)>| {
            if let Some((bin, row)) = at_row { rows.kill_at(w, bin, row); }
        };
        match best.entry(id) {
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

/// Raw user -> person name for one user bin. Each raw user recurs about a
/// dozen times per bin, and a hit here is predictable where the alias probe's
/// hit-or-miss outcome is not.
struct Names {
    table: Vec<(u64, u64)>,
    used: usize,
}

impl Names {
    fn reset(&mut self, rows: usize) {
        self.table.clear();
        // A raw user recurs about a dozen times per bin: start near 25% load.
        self.table.resize((rows / 4).max(64).next_power_of_two(), (0, 0));
        self.used = 0;
    }

    #[inline(always)]
    fn get(&mut self, key: u64, aliases: &Aliases) -> u64 {
        let mask = self.table.len() - 1;
        let mut i = hash(key) as usize & mask;
        loop {
            let (k, name) = self.table[i];
            if k == key { return name; }
            if k == 0 { return self.insert(i, key, aliases); }
            i = (i + 1) & mask;
        }
    }

    #[cold]
    #[inline(never)]
    fn insert(&mut self, i: usize, key: u64, aliases: &Aliases) -> u64 {
        let name = aliases.resolve(key);
        self.table[i] = (key, name);
        self.used += 1;
        if self.used * 2 > self.table.len() {
            let grown = vec![(0, 0); self.table.len() * 4];
            let old = std::mem::replace(&mut self.table, grown);
            let mask = self.table.len() - 1;
            for (k, v) in old.into_iter().filter(|e| e.0 != 0) {
                let mut j = hash(k) as usize & mask;
                while self.table[j].0 != 0 { j = (j + 1) & mask; }
                self.table[j] = (k, v);
            }
        }
        name
    }
}

/// Per-worker scratch for renaming.
#[derive(Default)]
struct Renamer {
    names: Option<Names>,
    ev: Vec<Event>,
    cl: Vec<Click>,
}

// Every raw user in this bin shares its top hash bits with exactly one alias
// shard, so these lookups stay in a cache-sized table. A user named by itself
// already lives in its person's bin: such rows are compacted in place and only
// renamed rows are scattered. Both sides are written unconditionally and only
// the cursors move, so the split costs no branch. Dedup losers (key 0) drop.
fn rename(mut parts: Vec<Histories>, aliases: &Aliases, dest: &mut [Histories], r: &mut Renamer, shift: u32, bin: usize) -> Vec<Histories> {
    let rows = parts.iter().map(|p| p.ev.len() + p.cl.len()).sum();
    let names = r.names.get_or_insert_with(|| Names { table: Vec::new(), used: 0 });
    names.reset(rows);
    for p in &mut parts {
        let n = p.ev.len();
        r.ev.clear();
        r.ev.reserve(n);
        let (mut kept, mut moved) = (0, 0);
        let stage = r.ev.as_mut_ptr();
        for i in 0..n {
            let e = p.ev[i];
            if e.key == 0 { continue; }
            let name = names.get(e.key, aliases);
            // SAFETY: kept <= i < n, and moved <= i < n <= stage capacity.
            unsafe {
                *p.ev.as_mut_ptr().add(kept) = e;
                *stage.add(moved) = Event { key: name, val: e.val };
            }
            kept += usize::from(name == e.key);
            moved += usize::from(name != e.key);
        }
        p.ev.truncate(kept);
        // SAFETY: the first `moved` staged rows were written above.
        unsafe { r.ev.set_len(moved) };
        for &e in &r.ev {
            dest[(hash(e.key) >> shift) as usize].ev.push(e);
        }
        let n = p.cl.len();
        r.cl.clear();
        r.cl.reserve(n);
        let (mut kept, mut moved) = (0, 0);
        let stage = r.cl.as_mut_ptr();
        for i in 0..n {
            let c = p.cl[i];
            let name = names.get(c.key, aliases);
            // SAFETY: as for events.
            unsafe {
                *p.cl.as_mut_ptr().add(kept) = c;
                *stage.add(moved) = Click { key: name, hit: c.hit };
            }
            kept += usize::from(name == c.key);
            moved += usize::from(name != c.key);
        }
        p.cl.truncate(kept);
        // SAFETY: the first `moved` staged rows were written above.
        unsafe { r.cl.set_len(moved) };
        for &c in &r.cl {
            dest[(hash(c.key) >> shift) as usize].cl.push(c);
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
    let events=Events{cx:&cx,clock:&clock,s:&shared,origin,b};
    let prep=total.elapsed();
    // Workers take chunks dynamically; leave headroom over an even share.
    let share=|bytes:usize,row:usize,n:usize|bytes/row/threads*5/4/n+64;
    let scans=thread::scope(|scope|{
        let handles:Vec<_>=(0..threads).map(|_|scope.spawn(||{
            let mut out=Scan{ev:sized(pb,share(event_bytes,77,pb)),cl:sized(pb,share(bytes[1],47,pb)),
                cand:sized(eb,share(event_bytes,77,eb)),wide:Vec::new(),links:Vec::new()};
            loop {
                let i=next.fetch_add(1,AtomicOrdering::Relaxed);
                let Some(c)=chunks.get(i) else{break};
                let buf=&maps[c.file as usize][c.start..c.end];
                match c.kind {
                    0=>scan_events(buf,&events,&mut out),
                    1=>scan_clicks(buf,&cx,&clock,&shared,&mut out,origin,b),
                    _=>wire::rows::<2>(buf,|[a,b]| {
                        if !a.is_empty() && !b.is_empty() && a!=b {
                            out.links.push((user_key(&shared,a),user_key(&shared,b)));
                        }
                    }),
                }
            }
            out
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).collect::<Vec<_>>()
    });
    let parsed=total.elapsed();
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
    let aliases=links::build(lks,&slow,threads,pb);
    let indexed=total.elapsed();
    let rows=Rows::new(&mut evs,b.user);
    consume(cands,threads,||(),|_,(c,wd)|dedup(c,wd,&rows,b.id));
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
        eprintln!("ms prep={} parse={} links={} dedup={} rename={} group={} merge={}",
            prep.as_millis(),(parsed-prep).as_millis(),(indexed-parsed).as_millis(),(reduced-indexed).as_millis(),
            (renamed-reduced).as_millis(),(grouped-renamed).as_millis(),(total.elapsed()-grouped).as_millis());
    }
    // The process exits right after writing; unmapping is wasted time.
    std::mem::forget(maps);
    std::mem::forget(aliases);
    Ok(Tables{users,camps_out,camps,cat_names,slow,organic})
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
                let v = payload(&f, &cx, &clock, at, origin).unwrap_or(0);
                acc ^= id ^ at as u64 ^ v;
            });
            std::hint::black_box(acc);
        });
        report("+user", &mut || {
            let mut acc = 0u64;
            wire::rows::<7>(body, |f| {
                let Some(id) = id_key(&shared, f[0]) else { return };
                let Some(at) = clock.ingest(f[3]) else { return };
                let v = payload(&f, &cx, &clock, at, origin).unwrap_or(0);
                let u = if f[1].is_empty() { 0 } else { user_key(&shared, f[1]) };
                acc ^= id ^ at as u64 ^ v ^ u;
            });
            std::hint::black_box(acc);
        });
        let b = Bits { user: 9, id: 11 };
        report("scan_events", &mut || {
            let mut out = Scan { ev: sized(1 << b.user, rows_n >> b.user << 1), cl: Vec::new(),
                                 cand: sized(1 << b.id, rows_n >> b.id << 1), wide: Vec::new(), links: Vec::new() };
            scan_events(body, &Events { cx: &cx, clock: &clock, s: &shared, origin, b }, &mut out);
            std::hint::black_box(&out.ev);
        });
        eprintln!("rows={rows_n} bytes={}", body.len());
    }
}
