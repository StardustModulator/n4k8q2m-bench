//! Shopping Sessions: exact CSV reduction and Parquet output.
mod aggregate;
mod atom;
mod clock;
#[cfg(test)]
mod fixture;
mod group;
mod links;
mod output;
mod stream;
mod time;
mod wire;

use atom::{is_slow, slow_index, slow_key, try_pack, write_id};
use memmap2::{Advice, Mmap};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::thread;
use std::time::Instant;
use stream::compute;
use time::{DAY_MS, WEEK_MS, format_day, parse_ts};
#[cfg(test)]
use time::GAP_MS;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Window { start: i64, end: i64, click_start: i64, start_day: i32 }
struct UserRow { key: u64, day: i32, sessions: i64, events: i64, revenue: i64, top: u16 }
struct CampRow { camp: u16, day: i32, sessions: i64, purchasing: i64, revenue: i64, assisted: i64 }
struct Camp { name: Vec<u8>, channel: Vec<u8> }
struct Tables {
    users: Vec<UserRow>, camps_out: Vec<CampRow>, camps: Vec<Camp>,
    cat_names: Vec<Vec<u8>>, slow: SlowIntern, organic: u16,
}
struct Ctx<'a> {
    prod_cat: &'a [u8], unknown: u8, prod_other: &'a HashMap<Vec<u8>, u8>,
    camp_by_num: &'a [i32], camp_other: &'a HashMap<Vec<u8>, u16>, window: &'a Window,
}

struct SlowIntern { map:HashMap<Vec<u8>,u32>, strings:Vec<Vec<u8>> }
impl SlowIntern {
    fn new()->Self {Self {map: HashMap::new(),strings:Vec::new()}}
    fn intern(&mut self,s:&[u8])->u64 {
        if let Some(&i)=self.map.get(s){return slow_key(i);}
        let i=u32::try_from(self.strings.len()).expect("too many distinct long IDs");
        self.strings.push(s.to_vec());
        self.map.insert(s.to_vec(),i);
        slow_key(i)
    }
}
struct Shared {slow:Mutex<SlowIntern>}
#[cold]
fn intern(s:&Shared,b:&[u8])->u64 {s.slow.lock().unwrap().intern(b)}
#[inline(always)]
fn user_key(s:&Shared,b:&[u8])->u64 {try_pack(b).unwrap_or_else(||intern(s,b))}
#[inline(always)]
fn id_key(s:&Shared,b:&[u8])->Option<u64>{(!b.is_empty()).then(|| user_key(s,b))}
#[inline]
fn with_key_bytes(key:u64, slow:&SlowIntern,mut f:impl FnMut(&[u8])) {
    if is_slow(key){f(&slow.strings[slow_index(key)]);}else{
        let mut buf=[0;24]; let n=write_id(key,&mut buf); f(&buf[..n]);
    }
}
#[inline]
fn cmp_keys(a:u64,b:u64,slow:&SlowIntern)->Ordering{
    if !is_slow(a) && !is_slow(b){return atom::packed_cmp(a,b);}
    cmp_interned(a,b,slow)
}
#[cold]
fn cmp_interned(a:u64,b:u64,slow:&SlowIntern)->Ordering{
    let mut res=Ordering::Equal;
    with_key_bytes(a,slow,|x| with_key_bytes(b,slow,|y|res=x.cmp(y)));
    res
}

struct ChunkLite {file:usize,kind:u8,start:usize,end:usize}
fn split_chunks(file:usize,kind:u8,b:&[u8])->Vec<ChunkLite>{
    let Some(header)=memchr::memchr(b'\n',b) else{return Vec::new()};
    let mut ret=Vec::new(); let mut start=header+1;
    while start<b.len(){
        let split=(start+1024*1024).min(b.len());
        let end=memchr::memchr(b'\n', &b[split..]).map_or(b.len(),|i|split+i+1);
        ret.push(ChunkLite{file,kind,start,end});start=end;
    }
    ret
}
fn list_csv(dir:&Path)->io::Result<Vec<PathBuf>>{
    let mut v=Vec::new();
    let iter=match fs::read_dir(dir){
        Ok(i)=>i, Err(e) if e.kind()==io::ErrorKind::NotFound=>return Ok(v), Err(e)=>return Err(e),
    };
    // Only regular files (through symlinks): a directory named *.csv is skipped.
    for ent in iter {let p=ent?.path();if p.extension().is_some_and(|e|e=="csv") && p.is_file(){v.push(p);}}
    v.sort_unstable();Ok(v)
}
fn load_window(path:&Path)->Result<Window,Box<dyn std::error::Error>>{
    let text=fs::read_to_string(path)?;
    let get=|field:&str|->Option<i64>{
        let (_,rest)=text.split_once(field)?;
        let (_,rest)=rest.split_once(':')?;let (_,rest)=rest.split_once('"')?;
        parse_ts(rest.split_once('"')?.0.as_bytes())
    };
    let start=get("\"window_start\"").ok_or("manifest: bad window_start")?;
    let end=get("\"window_end\"").ok_or("manifest: bad window_end")?;
    Ok(Window{start,end,click_start:start-WEEK_MS,start_day:start.div_euclid(DAY_MS) as i32})
}
#[inline]
fn u32_digits(s:&[u8])->Option<u32>{
    if s.is_empty(){return None;}
    let mut x=0u32;
    for &b in s{let d=b.wrapping_sub(b'0');if d>9{return None;}x=x.checked_mul(10)?.checked_add(u32::from(d))?;}
    Some(x)
}
#[inline]
fn canonical_number(s:&[u8])->Option<u32>{
    if s.len()>1 && s[0]==b'0'{return None;}
    if s.len()<=8 {
        if s.is_empty(){return None;}
        let mut n=0u32;
        for &b in s {let d=b.wrapping_sub(b'0');if d>9{return None;}n=n*10+u32::from(d);}
        Some(n)
    } else {u32_digits(s)}
}
#[inline]
fn parse_price(s:&[u8])->Option<u32>{u32_digits(s).filter(|p|(1..=10_000_000).contains(p))}
#[inline]
fn product_cat(c:&Ctx<'_>,s:&[u8])->u8{
    if let Some(n)=s.strip_prefix(b"p").and_then(canonical_number) {
        if let Some(&v)=c.prod_cat.get(n as usize){return if v==255{c.unknown}else{v};}
    }
    c.prod_other.get(s).copied().unwrap_or(c.unknown)
}
#[inline]
fn campaign_of(c:&Ctx<'_>,s:&[u8])->Option<u16>{
    if let Some(n)=s.strip_prefix(b"c_").and_then(canonical_number){
        if let Some(&v)=c.camp_by_num.get(n as usize){return (v>=0).then_some(v as u16);}
    }
    c.camp_other.get(s).copied()
}
fn body(b:&[u8])->&[u8]{memchr::memchr(b'\n',b).map_or(&[],|i|&b[i+1..])}
type Products=(Vec<u8>,u8,HashMap<Vec<u8>,u8>,Vec<Vec<u8>>);
fn load_products(path:&Path)->Result<Products,Box<dyn std::error::Error>>{
    let b=fs::read(path)?;
    let mut names:Vec<Vec<u8>>=vec![b"unknown".to_vec()];
    let mut dense=Vec::new();let mut other=HashMap::new();let mut too_many=false;
    wire::rows::<2>(body(&b),|[id,name]|{
        if id.is_empty(){return;}
        let cat=if let Some(i)=names.iter().position(|c|c==name){i}else{
            if names.len()>=255 {too_many=true;return;}
            names.push(name.to_vec());names.len()-1
        } as u8;
        if let Some(n)=id.strip_prefix(b"p").and_then(canonical_number).map(|n|n as usize).filter(|&n|n<=b.len()){
            if n>=dense.len(){dense.resize(n+1,255);}
            dense[n]=cat;
        }else{other.insert(id.to_vec(),cat);}
    });
    if too_many{return Err("product category count exceeds compact payload".into());}
    Ok((dense,0,other,names))
}
type Campaigns=(Vec<i32>,Vec<Camp>,HashMap<Vec<u8>,u16>);
fn load_campaigns(path:&Path)->Result<Campaigns,Box<dyn std::error::Error>>{
    let b=fs::read(path)?;
    let mut dense=Vec::new();let mut camps=Vec::new();let mut other=HashMap::new();let mut too_many=false;
    wire::rows::<2>(body(&b),|[id,chan]|{
        if id.is_empty(){return;}
        if camps.len()>=u16::MAX as usize{too_many=true;return;}
        let i=camps.len() as u16;
        if let Some(n)=id.strip_prefix(b"c_").and_then(canonical_number).map(|n|n as usize).filter(|&n|n<=b.len()){
            if n>=dense.len(){dense.resize(n+1,-1);}
            dense[n]=i32::from(i);
        }else{other.insert(id.to_vec(),i);}
        camps.push(Camp{name:id.to_vec(),channel:chan.to_vec()});
    });
    if too_many{return Err("too many campaign IDs".into());}
    Ok((dense,camps,other))
}

fn main() {
    let input=std::env::var_os("DATA_DIR").unwrap_or_else(||"/data".into());
    let out=std::env::var_os("OUTPUT_DIR").unwrap_or_else(||"/output".into());
    let threads=std::env::var("NCPU").ok().and_then(|s|s.parse::<usize>().ok()).filter(|&n|n>0)
        .unwrap_or_else(||thread::available_parallelism().map_or(1,usize::from));
    if let Err(e)=run(Path::new(&input),Path::new(&out),threads){eprintln!("solve: {e}");std::process::exit(1);}
    // Skip destructors: freeing gigabytes is pure overhead at exit.
    std::process::exit(0);
}
fn run(data:&Path,out:&Path,threads:usize)->Result<(),Box<dyn std::error::Error>>{
    let t=Instant::now();
    let tables=compute(data,threads)?;
    let t1=Instant::now();
    output::write(out,&tables,threads)?;
    if std::env::var_os("WRITE_CSV").is_some(){write_debug_csv(out,&tables)?;}
    eprintln!("rows user={} camp={} compute_ms={} write_ms={} total_ms={}",tables.users.len(),tables.camps_out.len(),
        (t1-t).as_millis(),t1.elapsed().as_millis(),t.elapsed().as_millis());
    std::mem::forget(tables);
    Ok(())
}
fn write_debug_csv(out:&Path,t:&Tables)->io::Result<()>{
    let mut u=io::BufWriter::new(File::create(out.join("user_daily_debug.csv"))?);
    writeln!(u,"user_id,day,sessions,events,revenue_cents,top_category")?;
    let mut date=[0;10];
    for r in &t.users{
        let mut ok=Ok(());
        with_key_bytes(r.key,&t.slow,|bytes|ok=u.write_all(bytes));ok?;
        format_day(r.day,&mut date);
        write!(u,",{},{},{},{},",std::str::from_utf8(&date).unwrap(),r.sessions,r.events,r.revenue)?;
        if r.top!=u16::MAX{u.write_all(&t.cat_names[r.top as usize])?;}
        u.write_all(b"\n")?;
    }
    u.flush()?;
    let mut c=io::BufWriter::new(File::create(out.join("campaign_daily_debug.csv"))?);
    writeln!(c,"campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions")?;
    for r in &t.camps_out{
        let k=&t.camps[r.camp as usize];
        c.write_all(&k.name)?;c.write_all(b",")?;c.write_all(&k.channel)?;format_day(r.day,&mut date);
        writeln!(c,",{},{},{},{},{}",std::str::from_utf8(&date).unwrap(),r.sessions,r.purchasing,r.revenue,r.assisted)?;
    }
    c.flush()
}
