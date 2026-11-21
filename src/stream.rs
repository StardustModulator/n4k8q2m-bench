//! Partition on event ID *before* dropping invalid records. Partition on the
//! linked person's name *after* choosing the earliest copy. These two shuffles
//! are sufficient; neither an all-user interner nor a global sort is needed.
use super::*;
use crate::aggregate::{Acc, Hit};
use crate::group::{Click, Event, Histories};
use crate::clock::Clock;
use crate::links::Aliases;

#[inline(always)]
pub(super) fn hash(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[derive(Clone, Copy)]
struct Record {
    id: u64,
    user: u64, // zero is invalid; retained in the minimum-ingest contest
    at: i64,
    val: u64,  // 32-bit timestamp, 8-bit category, 24-bit price
}

pub(super) fn bins<T>(n: usize) -> Vec<Vec<T>> {
    (0..n).map(|_| Vec::new()).collect()
}

struct Scan {
    ev: Vec<Vec<Record>>,
    cl: Vec<Click>,
    links: Vec<(u64, u64)>,
}

#[inline(always)]
fn payload(f: &[&[u8]; 7], cx: &Ctx<'_>, clock: &Clock, at: i64, origin: i64) -> Option<u64> {
    let ts = clock.parse(f[2])?;
    if ts < cx.window.start || ts >= cx.window.end || at > ts + DAY_MS || f[5].is_empty() {
        return None;
    }
    let pay = match f[4] {
        b"view" | b"cart" => 0,
        b"purchase" => {
            let p = parse_price(f[6])?;
            p | (u32::from(product_cat(cx, f[5])) << 24)
        }
        _ => return None,
    };
    Some(((ts - origin) as u64) << 32 | u64::from(pay))
}

#[inline]
fn scan_events(buf: &[u8], cx: &Ctx<'_>, clock: &Clock, s: &Shared, out: &mut Scan, origin: i64) {
    let mask = out.ev.len() - 1;
    wire::rows::<7>(buf, |f| {
        let Some(id) = id_key(s, f[0]) else { return };
        let Some(at) = clock.ingest(f[3]) else { return };
        let (user, val) = if f[1].is_empty() {
            (0, 0)
        } else if let Some(v) = payload(&f, cx, clock, at, origin) {
            (user_key(s, f[1]), v)
        } else {
            (0, 0)
        };
        let b = (hash(id) >> 32) as usize & mask;
        out.ev[b].push(Record { id, user, at, val });
    });
}

#[inline]
fn scan_clicks(buf: &[u8], cx: &Ctx<'_>, clock: &Clock, s: &Shared, out: &mut Scan, origin: i64) {
    out.cl.reserve(buf.len() / 45);
    wire::rows::<4>(buf, |f| {
        if f[0].is_empty() || f[1].is_empty() { return; }
        let Some(ts) = clock.parse(f[3]) else { return };
        if ts < cx.window.click_start || ts >= cx.window.end { return; }
        let Some(camp) = campaign_of(cx, f[2]) else { return };
        let hit = Hit { ts: (ts - origin) as u32, id: user_key(s, f[0]), camp };
        out.cl.push(Click { key: user_key(s, f[1]), hit });
    });
}

// Copies of one id may sit in different workers' parts. The parts are read
// in place; concatenating them would copy every record into fresh pages.
fn reduce_event_ids(mut parts: Vec<Vec<Record>>, dest: &mut [Histories], shift: u32) {
    let n: usize = parts.iter().map(Vec::len).sum();
    if n == 0 { return; }
    // Two bits per hash prefix. A prefix seen once cannot be a duplicate, so
    // those rows skip the id table. A shared prefix is only a suspect: the
    // exact table still compares full ids.
    let words = n.div_ceil(4).next_power_of_two();
    let word_mask = words - 1;
    let mut present = vec![0u64; words];
    let mut repeated = vec![0u64; words];
    for r in parts.iter().flatten() {
        let h = hash(r.id);
        let bit = 1u64 << (h & 63);
        let i = ((h >> 6) as usize) & word_mask;
        let seen = present[i];
        repeated[i] |= seen & bit;
        present[i] = seen | bit;
    }
    let mut suspects = n;
    for i in 0..words {
        suspects -= (present[i] ^ repeated[i]).count_ones() as usize;
    }
    if suspects > 0 {
        // Slot value: part index + 1 in the high half, row in the low half.
        let mask = (suspects * 2).next_power_of_two() - 1;
        let mut table = vec![0u64; mask + 1];
        for p in 0..parts.len() {
            for i in 0..parts[p].len() {
                let id = parts[p][i].id;
                let h = hash(id);
                if ((repeated[((h >> 6) as usize) & word_mask] >> (h & 63)) & 1) == 0 {
                    continue;
                }
                let me = ((p as u64 + 1) << 32) | i as u64;
                let mut s = hash(id.wrapping_mul(0x9e37_79b9_7f4a_7c15)) as usize & mask;
                loop {
                    let ptr = table[s];
                    if ptr == 0 {
                        table[s] = me;
                        break;
                    }
                    let (q, j) = ((ptr >> 32) as usize - 1, ptr as u32 as usize);
                    if parts[q][j].id == id {
                        if parts[p][i].at < parts[q][j].at {
                            parts[q][j].user = 0;
                            table[s] = me;
                        } else {
                            parts[p][i].user = 0;
                        }
                        break;
                    }
                    s = (s + 1) & mask;
                }
            }
        }
    }
    for r in parts.into_iter().flatten() {
        if r.user != 0 {
            dest[(hash(r.user) >> shift) as usize].ev.push(Event { key: r.user, val: r.val });
        }
    }
}

fn route_clicks(rows: Vec<Click>, dest: &mut [Histories], shift: u32) {
    // All copies of a click are identical. Window membership is existential
    // per campaign and credit is a min/max over keys. Both are idempotent:
    // duplicating an identical tuple cannot change either result.
    for r in rows {
        dest[(hash(r.key) >> shift) as usize].cl.push(r);
    }
}

// Every raw user in this bin shares its top hash bits with exactly one alias
// shard, so these lookups stay in a cache-sized table. A user named by itself
// already lives in its person's bin: such rows are compacted in place and only
// renamed rows are scattered.
fn rename(mut parts: Vec<Histories>, aliases: &Aliases, dest: &mut [Histories], shift: u32, bin: usize) -> Vec<Histories> {
    for p in &mut parts {
        let mut kept = 0;
        for i in 0..p.ev.len() {
            let mut e = p.ev[i];
            let name = aliases.resolve(e.key);
            if name == e.key {
                p.ev[kept] = e;
                kept += 1;
            } else {
                e.key = name;
                dest[(hash(name) >> shift) as usize].ev.push(e);
            }
        }
        p.ev.truncate(kept);
        let mut kept = 0;
        for i in 0..p.cl.len() {
            let mut c = p.cl[i];
            let name = aliases.resolve(c.key);
            if name == c.key {
                p.cl[kept] = c;
                kept += 1;
            } else {
                c.key = name;
                dest[(hash(name) >> shift) as usize].cl.push(c);
            }
        }
        p.cl.truncate(kept);
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

enum Route { Ev(Vec<Vec<Record>>), Cl(Vec<Click>) }

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
    let mut event_bytes=0;
    for (kind,dir) in [(0u8,"events"),(1,"clicks"),(2,"links")] {
        for p in list_csv(&data.join(dir))? {
            let f=File::open(p)?;
            if f.metadata()?.len()==0 {continue;}
            let m=unsafe{Mmap::map(&f)?};
            let _=m.advise(Advice::Sequential);
            if kind==0 {event_bytes+=m.len();}
            chunks.extend(split_chunks(maps.len(),kind,&m));
            maps.push(m);
        }
    }
    let threads=threads.max(1).min(chunks.len().max(1));
    let eb=(event_bytes / (77 * 16384)).next_power_of_two().clamp(8,2048);
    let pb=(event_bytes / (77 * 65536)).next_power_of_two().clamp(16,512);
    chunks.sort_unstable_by_key(|c|std::cmp::Reverse(c.end-c.start));
    let next=AtomicUsize::new(0);
    let shared=Shared{slow:Mutex::new(SlowIntern::new())};
    let prep=total.elapsed();
    let scans=thread::scope(|scope|{
        let handles:Vec<_>=(0..threads).map(|_|scope.spawn(||{
            let per_bin=event_bytes/(70*eb*threads)+64;
            let mut out=Scan{ev:(0..eb).map(|_|Vec::with_capacity(per_bin)).collect(),cl:Vec::new(),links:Vec::new()};
            loop {
                let i=next.fetch_add(1,AtomicOrdering::Relaxed);
                let Some(c)=chunks.get(i) else{break};
                let buf=&maps[c.file as usize][c.start..c.end];
                match c.kind {
                    0=>scan_events(buf,&cx,&clock,&shared,&mut out,origin),
                    1=>scan_clicks(buf,&cx,&clock,&shared,&mut out,origin),
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
    let mut parts:Vec<Vec<Vec<Record>>>=bins(eb);
    let mut jobs=Vec::with_capacity(threads+eb);
    let mut lks=Vec::with_capacity(threads);
    for s in scans {
        for (b,v) in parts.iter_mut().zip(s.ev) {b.push(v);}
        if !s.cl.is_empty(){jobs.push(Route::Cl(s.cl));}
        lks.push(s.links);
    }
    let aliases=links::build(lks,&slow,threads,pb);
    let indexed=total.elapsed();
    jobs.extend(parts.into_iter().map(Route::Ev));
    let shift=64-pb.trailing_zeros();
    let hist=||(0..pb).map(|_| Histories::default()).collect::<Vec<_>>();
    let routed=consume(jobs,threads,hist,
        |out,j|match j{
            Route::Ev(rows)=>reduce_event_ids(rows,out,shift),
            Route::Cl(rows)=>route_clicks(rows,out,shift),
        });
    let reduced=total.elapsed();
    let mut by_user:Vec<Vec<Histories>>=bins(pb);
    for r in routed {
        for (dst,src) in by_user.iter_mut().zip(r) {dst.push(src);}
    }
    let renamed=total.elapsed(); // rename now happens inside persons
    let partials=crate::persons::run(by_user,&aliases,threads,shift,||Acc::new(camps.len(),days,cat_names.len()),
                                     &cat_names,&slow,w.start_day);
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
        eprintln!("ms prep={} parse={} links={} dedup+route+rename={} histories={} merge={}",
            prep.as_millis(),(parsed-prep).as_millis(),(indexed-parsed).as_millis(),(reduced-indexed).as_millis(),
            (grouped-reduced).as_millis(),(total.elapsed()-grouped).as_millis());
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
        let report = |name: &str, f: &mut dyn FnMut()| {
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
        report("scan_events", &mut || {
            let mut out = Scan { ev: (0..2048).map(|_| Vec::with_capacity(rows_n / 2048 + 64)).collect(), cl: Vec::new(), links: Vec::new() };
            scan_events(body, &cx, &clock, &shared, &mut out, origin);
            std::hint::black_box(&out.ev);
        });
        eprintln!("rows={rows_n} bytes={}", body.len());
    }
}
