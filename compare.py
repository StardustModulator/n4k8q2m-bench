import sys, csv
got_dir, ans_dir = sys.argv[1], sys.argv[2]
ok = True
for t in ("user_daily", "campaign_daily"):
    got = set(map(tuple, list(csv.reader(open(f"{got_dir}/{t}_debug.csv")))[1:]))
    want = set(map(tuple, list(csv.reader(open(f"{ans_dir}/{t}.csv")))[1:]))
    print(t, "miss", len(want - got), "extra", len(got - want))
    ok &= got == want
sys.exit(0 if ok else 1)
