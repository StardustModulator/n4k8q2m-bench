use super::*;
use time::days_from_civil;
    use std::collections::BTreeSet;

    fn day(y: i64, m: i64, d: i64) -> i32 {
        days_from_civil(y, m, d).unwrap() as i32
    }

    #[test]
    fn fixture_matches_hand_rules() {
        let root = std::env::temp_dir().join(format!("ss-fix-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("events")).unwrap();
        fs::create_dir_all(root.join("clicks")).unwrap();
        fs::create_dir_all(root.join("links")).unwrap();
        fs::write(
            root.join("manifest.json"),
            "{\n  \"window_start\": \"2026-01-01T00:00:00Z\",\n  \"window_end\": \"2026-01-15T00:00:00Z\"\n}\n",
        )
        .unwrap();
        fs::write(root.join("products.csv"), "product_id,category\np1,books\np2,toys\n").unwrap();
        fs::write(root.join("campaigns.csv"), "campaign_id,channel\nc_1,search\nc_2,email\n").unwrap();
        fs::write(
            root.join("links").join("part.csv"),
            "user_id_a,user_id_b\nu2,u1\nu9,u9\nu2,U2\nu2,u1\nu1,\nonly_one\n",
        )
        .unwrap();
        let t_jan9_30 = parse_ts(b"2026-01-09T00:30:00Z").unwrap();
        let f3_split = (t_jan9_30 + GAP_MS + 1).to_string();
        let t_jan1 = parse_ts(b"2026-01-01T00:00:00Z").unwrap();
        let early = (t_jan1 - 1).to_string();
        let t_jan8 = parse_ts(b"2026-01-08T00:00:00Z").unwrap();
        let after = (t_jan8 + 1).to_string();
        let events = format!(
            "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n\
e1,u2,2026-01-01T00:00:00Z,2026-01-01T00:00:01Z,purchase,p1,0250\n\
e2,u2,2026-01-01T00:10:00Z,2026-01-01T00:10:01Z,view,p1,\n\
e3,u1,2026-01-01T00:30:00Z,2026-01-01T00:30:01Z,purchase,p2,100\n\
e4,u1,2026-01-01T23:50:00Z,2026-01-01T23:50:01Z,view,p1,\n\
e5,u2,2026-01-02T00:05:00Z,2026-01-02T00:05:01Z,view,p1,\n\
e6,u3,2026-01-03T00:00:00Z,2026-01-03T00:00:01Z,view,p999,\n\
e12,u3,2026-01-03T01:00:00Z,2026-01-04T01:00:01Z,view,p1,\n\
e13,u3,2026-01-03T02:00:00Z,2026-01-04T02:00:00Z,view,p1,\n\
e20,u3,not-a-ts,2026-01-03T03:00:01Z,view,p1,\n\
e20,u3,2026-01-03T03:00:00Z,2026-01-03T03:05:00Z,view,p1,\n\
e21,u3,2026-01-05T00:00:00Z,2026-01-05T00:00:01Z,purchase,p1,100\n\
e21,u3,2026-01-05T00:00:00Z,2026-01-05T00:10:00Z,purchase,p1,999\n\
e22,u3,2026-01-05T00:00:00Z\n\
\n\
e23,\"u3,2026-01-05T01:00:00Z,2026-01-05T01:00:01Z,view,p1,\n\
e80,u3,2026-01-05T05:00:00Z,2026-01-05T05:00:01Z,purchase,p1,0\n\
e81,u3,2026-01-05T06:00:00Z,2026-01-05T06:00:01Z,Purchase,p1,100\n\
e10,u4,2026-01-04T00:00:00Z,2026-01-04T00:00:01Z,purchase,p999,100\n\
e11,u4,2026-01-04T01:00:00Z,2026-01-04T01:00:01Z,view,p1,\n\
e40,u5,2026-01-08T00:00:00Z,2026-01-08T00:00:01Z,view,p1,\n\
e50,u6,2026-01-07T00:00:00Z,2026-01-07T00:00:01Z,purchase,p1,100\n\
e51,u6,2026-01-07T00:10:00Z,2026-01-07T00:10:01Z,purchase,p2,100\n\
e60,u7,2026-01-09T00:00:00Z,2026-01-09T00:00:01Z,view,p1,\n\
e61,u7,2026-01-09T00:30:00Z,2026-01-09T00:30:01Z,view,p1,\n\
e62,u7,{f3_split},2026-01-09T01:00:01Z,view,p1,\n\
e70,u8,2026-01-10T00:00:00Z,2026-01-10T00:00:01Z,view,p1,\n\
\"e30\",\"u9\",\"2026-01-06T00:00:00Z\",\"2026-01-06T00:00:01Z\",\"view\",\"p1\",\"\"\n\
e31,u9,2026-01-06T00:00:00Z,2026-01-06T00:00:02Z,view,p1,\n\
e92,u10,2026-01-11T00:00:00Z,2026-01-11T00:00:01Z,purchase,P1,50\n\
e90,u11,2026-01-12T03:30:00-04:00,2026-01-12T07:30:01Z,view,p1,\n\
e91,u11,2026-01-12T22:30:00-04:00,2026-01-13T02:30:01Z,view,p1,\n"
        );
        fs::write(root.join("events").join("a.csv"), events).unwrap();
        fs::write(root.join("events").join("empty.csv"), "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents\n").unwrap();
        let clicks = format!(
            "click_id,user_id,campaign_id,ts\n\
k1,u4,c_1,2026-01-04T00:00:00Z\n\
\"k1\",\"u4\",\"c_1\",\"2026-01-04T00:00:00Z\"\n\
k9,u4,c_2,2026-01-03T00:00:00Z\n\
k7,u5,c_1,2026-01-01T00:00:00Z\n\
k8,u5,c_2,{early}\n\
k6,u5,c_2,{after}\n\
k2,u8,c_2,2026-01-09T00:00:00Z\n\
k1b,u8,c_1,2026-01-09T00:00:00Z\n\
kbad,u8,c_9999,2026-01-09T00:00:00Z\n"
        );
        // k1 and k1b: k1 < k1b? "k1" vs "k1b": k1 is a prefix, so k1 < k1b.
        // I wanted k1 to win against k2. k1 is on u4, not u8.
        // For u8 the tie is k2 (c_2) and k1b (c_1). "k1b" < "k2" because '1' < '2'.
        // Credit c_1. Good, k1b wins.
        fs::write(root.join("clicks").join("a.csv"), clicks).unwrap();

        let tables = compute(&root, 4).unwrap();
        let mut got_u = BTreeSet::new();
        for row in &tables.users {
            let mut name = Vec::new();
            with_key_bytes(row.key, &tables.slow, |b| name.extend_from_slice(b));
            let top = if row.top == u16::MAX {
                String::new()
            } else {
                String::from_utf8_lossy(&tables.cat_names[row.top as usize]).into_owned()
            };
            got_u.insert((
                String::from_utf8(name).unwrap(),
                row.day,
                row.sessions,
                row.events,
                row.revenue,
                top,
            ));
        }
        let mut expect_u = BTreeSet::new();
        let push = |set: &mut BTreeSet<_>, name: &str, y, m, d, s, e, r, top: &str| {
            set.insert((name.into(), day(y, m, d), s, e, r, top.into()));
        };
        push(&mut expect_u, "U2", 2026, 1, 1, 2, 5, 350, "books");
        push(&mut expect_u, "u3", 2026, 1, 3, 2, 2, 0, "");
        push(&mut expect_u, "u3", 2026, 1, 5, 1, 1, 100, "books");
        push(&mut expect_u, "u4", 2026, 1, 4, 2, 2, 100, "unknown");
        push(&mut expect_u, "u5", 2026, 1, 8, 1, 1, 0, "");
        push(&mut expect_u, "u6", 2026, 1, 7, 1, 2, 200, "books");
        push(&mut expect_u, "u7", 2026, 1, 9, 2, 3, 0, "");
        push(&mut expect_u, "u8", 2026, 1, 10, 1, 1, 0, "");
        push(&mut expect_u, "u9", 2026, 1, 6, 1, 2, 0, "");
        push(&mut expect_u, "u10", 2026, 1, 11, 1, 1, 50, "unknown");
        push(&mut expect_u, "u11", 2026, 1, 12, 1, 1, 0, "");
        push(&mut expect_u, "u11", 2026, 1, 13, 1, 1, 0, "");
        assert_eq!(got_u, expect_u, "users got={got_u:?}");

        let mut got_c = BTreeSet::new();
        for row in &tables.camps_out {
            got_c.insert((
                String::from_utf8_lossy(&tables.camps[row.camp as usize].name).into_owned(),
                String::from_utf8_lossy(&tables.camps[row.camp as usize].channel).into_owned(),
                row.day,
                row.sessions,
                row.purchasing,
                row.revenue,
                row.assisted,
            ));
        }
        let mut expect_c = BTreeSet::new();
        let pushc = |set: &mut BTreeSet<_>, id: &str, ch: &str, y, m, d, s, p, r, a| {
            set.insert((id.into(), ch.into(), day(y, m, d), s, p, r, a));
        };
        // organic: U2 (2 sess, 1 purch, 350), u3 jan3 (2,0,0), u3 jan5 (1,1,100),
        // u6 (1,1,200), u7 (2,0,0), u9 (1,0,0), u10 (1,1,50), u11 two days (1,0,0) each.
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 1, 2, 1, 350, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 3, 2, 0, 0, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 5, 1, 1, 100, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 6, 1, 0, 0, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 7, 1, 1, 200, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 9, 2, 0, 0, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 11, 1, 1, 50, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 12, 1, 0, 0, 0);
        pushc(&mut expect_c, "organic", "organic", 2026, 1, 13, 1, 0, 0, 0);
        // c_1: u4 both sessions, u5, u8
        pushc(&mut expect_c, "c_1", "search", 2026, 1, 4, 2, 1, 100, 2);
        pushc(&mut expect_c, "c_1", "search", 2026, 1, 8, 1, 0, 0, 1);
        pushc(&mut expect_c, "c_1", "search", 2026, 1, 10, 1, 0, 0, 1);
        // c_2 assists u4 both sessions and u8
        pushc(&mut expect_c, "c_2", "email", 2026, 1, 4, 0, 0, 0, 2);
        pushc(&mut expect_c, "c_2", "email", 2026, 1, 10, 0, 0, 0, 1);
        assert_eq!(got_c, expect_c, "camps got={got_c:?}");
        let _ = fs::remove_dir_all(&root);
    }
