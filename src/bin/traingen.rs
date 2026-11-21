//! Writes a small dataset in the input format, for profile-guided builds.
//! Output values are made up; only the shape of the text matters.
use std::fs::{self, File};
use std::io::{BufWriter, Write};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
    fn chance(&mut self, permille: u64) -> bool { self.below(1000) < permille }
}

const DAY: i64 = 86_400_000;
const START: i64 = 1_800_000_000_000; // 2027-01-15T08:00:00Z, rounded below to midnight
const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

fn base36(mut v: u64, out: &mut Vec<u8>) {
    let mut buf = [0u8; 13];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = DIGITS[(v % 36) as usize];
        v /= 36;
        if v == 0 { break; }
    }
    out.extend_from_slice(&buf[i..]);
}

fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn iso(ms: i64, out: &mut Vec<u8>) {
    let (y, m, d) = civil(ms.div_euclid(DAY));
    let s = ms.rem_euclid(DAY) / 1000;
    out.extend_from_slice(format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60).as_bytes());
}

fn ts(rng: &mut Rng, ms: i64, out: &mut Vec<u8>) {
    match rng.below(100) {
        0..=57 => { iso(ms - ms.rem_euclid(1000), out); out.push(b'Z'); }
        58..=67 => {
            let off = rng.below(14) as i64 * 30 * 60_000;
            let sign = if rng.below(2) == 0 { b'+' } else { b'-' };
            let local = if sign == b'+' { ms + off } else { ms - off };
            iso(local - local.rem_euclid(1000), out);
            out.push(sign);
            out.extend_from_slice(format!("{:02}:{:02}", off / 3_600_000, off / 60_000 % 60).as_bytes());
        }
        _ => out.extend_from_slice(ms.to_string().as_bytes()),
    }
}

fn field(rng: &mut Rng, quoted_permille: u64, body: &[u8], out: &mut Vec<u8>) {
    let q = rng.chance(quoted_permille);
    if q { out.push(b'"'); }
    out.extend_from_slice(body);
    if q { out.push(b'"'); }
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: traingen DIR [events]");
    let events: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1_500_000);
    let start = START - START.rem_euclid(DAY);
    let end = start + 14 * DAY;
    for d in ["events", "clicks", "links"] { fs::create_dir_all(format!("{dir}/{d}")).unwrap(); }
    fs::write(format!("{dir}/manifest.json"), {
        let mut a = Vec::new(); let mut b = Vec::new();
        iso(start, &mut a); iso(end, &mut b);
        format!("{{\n  \"window_start\": \"{}Z\",\n  \"window_end\": \"{}Z\"\n}}\n", String::from_utf8(a).unwrap(), String::from_utf8(b).unwrap())
    }).unwrap();
    let cats = ["books", "garden", "toys", "music", "games", "home", "shoes", "pets", "tools", "food",
                "sport", "baby", "auto", "art", "beauty", "office", "phone", "video", "watch", "travel"];
    let mut w = BufWriter::new(File::create(format!("{dir}/products.csv")).unwrap());
    writeln!(w, "product_id,category").unwrap();
    for p in 1..=50_000u64 { writeln!(w, "p{p},{}", cats[(p * 7919 % 20) as usize]).unwrap(); }
    w.flush().unwrap();
    let chans = ["email", "social", "search", "display", "affiliate"];
    let mut w = BufWriter::new(File::create(format!("{dir}/campaigns.csv")).unwrap());
    writeln!(w, "campaign_id,channel").unwrap();
    for c in 1..=2000u64 { writeln!(w, "c_{c},{}", chans[(c * 31 % 5) as usize]).unwrap(); }
    w.flush().unwrap();

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let users = (events / 30).max(1000);
    let user = |rng: &mut Rng, out: &mut Vec<u8>| {
        // Zipf-like: a few users carry many rows.
        let r = rng.below(1000);
        let u = if r < 20 { rng.below(5) } else if r < 300 { rng.below(users / 100 + 1) } else { rng.below(users) };
        out.push(b'u');
        base36(u * 48_271 % (1 << 40) + 1, out);
    };
    let mut line = Vec::with_capacity(160);
    let files = 4;
    for f in 0..files {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(format!("{dir}/events/part-{f:05}.csv")).unwrap());
        w.write_all(b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n").unwrap();
        for i in 0..events / files {
            line.clear();
            let id = (f * events + i) * 2_654_435_761 % (1 << 40) + 1;
            let t = start + rng.below((14 * DAY) as u64) as i64 - if rng.chance(3) { 8 * DAY } else { 0 };
            let copies = if rng.chance(30) { 1 + rng.below(3) } else { 1 };
            for c in 0..copies {
                line.clear();
                if rng.chance(1) { line.extend_from_slice(b"garbage line without enough fields\n"); w.write_all(&line).unwrap(); continue; }
                let mut b = Vec::new(); b.push(b'e'); base36(id, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b',');
                if rng.chance(2) { field(&mut rng, 200, b"", &mut line); } else { let mut b = Vec::new(); user(&mut rng, &mut b); field(&mut rng, 200, &b, &mut line); }
                line.push(b',');
                let mut b = Vec::new();
                if rng.chance(5) { b.extend_from_slice(b"2027-13-40T25:61:61Z"); } else { ts(&mut rng, t, &mut b); }
                field(&mut rng, 200, &b, &mut line); line.push(b',');
                let late = if rng.chance(10) { DAY + rng.below(4 * DAY as u64) as i64 } else { rng.below(600_000) as i64 };
                let ing = t + late + c as i64 * 1000;
                let mut b = Vec::new(); iso(ing - ing.rem_euclid(1000), &mut b); b.push(b'Z'); field(&mut rng, 200, &b, &mut line); line.push(b',');
                let kind = rng.below(1000);
                let typ: &[u8] = if kind < 30 { b"purchase" } else if kind < 200 { b"cart" } else if kind < 999 { b"view" } else { b"View" };
                field(&mut rng, 200, typ, &mut line); line.push(b',');
                let mut b = Vec::new(); if rng.chance(5) { b.extend_from_slice(b"p99999999"); } else { b.push(b'p'); b.extend_from_slice((rng.below(50_000) + 1).to_string().as_bytes()); }
                field(&mut rng, 200, &b, &mut line); line.push(b',');
                let mut b = Vec::new();
                if typ == b"purchase" { if rng.chance(50) { b.extend_from_slice(b"-5"); } else { b.extend_from_slice((rng.below(500_000) + 1).to_string().as_bytes()); } }
                else if rng.chance(300) { b.extend_from_slice(b"0"); }
                field(&mut rng, 200, &b, &mut line); line.push(b'\n');
                w.write_all(&line).unwrap();
            }
        }
        w.flush().unwrap();
    }
    let clicks = events / 4;
    for f in 0..files {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(format!("{dir}/clicks/part-{f:05}.csv")).unwrap());
        w.write_all(b"click_id,user_id,campaign_id,ts\n").unwrap();
        for i in 0..clicks / files {
            line.clear();
            let id = (f * clicks + i) * 2_246_822_519 % (1 << 40) + 1;
            let mut b = Vec::new(); b.push(b'k'); base36(id, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b',');
            let mut b = Vec::new(); user(&mut rng, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b',');
            let mut b = Vec::new(); b.extend_from_slice(b"c_"); let c = if rng.chance(300) { rng.below(5) + 1 } else { rng.below(2000) + 1 }; b.extend_from_slice((c + if rng.chance(5) { 5000 } else { 0 }).to_string().as_bytes());
            field(&mut rng, 200, &b, &mut line); line.push(b',');
            let t = start - 7 * DAY + rng.below((21 * DAY) as u64) as i64 - if rng.chance(5) { 30 * DAY } else { 0 };
            let mut b = Vec::new(); ts(&mut rng, t, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b'\n');
            w.write_all(&line).unwrap();
            if rng.chance(20) { w.write_all(&line).unwrap(); }
        }
        w.flush().unwrap();
    }
    let mut w = BufWriter::new(File::create(format!("{dir}/links/part-00000.csv")).unwrap());
    w.write_all(b"user_id_a,user_id_b\n").unwrap();
    for _ in 0..events / 150 {
        line.clear();
        let mut b = Vec::new(); user(&mut rng, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b',');
        let mut b = Vec::new(); user(&mut rng, &mut b); field(&mut rng, 200, &b, &mut line); line.push(b'\n');
        w.write_all(&line).unwrap();
    }
    w.flush().unwrap();
}
