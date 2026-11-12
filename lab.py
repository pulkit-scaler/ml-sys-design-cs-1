#!/usr/bin/env python3
"""Feed lab driver. Python 3.9+ standard library only, so nothing to install.

    python3 lab.py status                 is everything up, and where to look
    python3 lab.py trace [--user N]       one feed request, hop by hop, against the 200 ms budget
    python3 lab.py sweep                  ANN recall and search time for several nprobe values
    python3 lab.py nprobe N               change nprobe on the running retrieval service
    python3 lab.py load [--rps R] [--seconds S]   steady traffic, one line per 2 seconds
    python3 lab.py fresh [--user N]       hide a topic, watch the next page change
    python3 lab.py page [--user N]        the user's next page and session features, nothing reset
    python3 lab.py drill                  ranker healthy, stopped, then hung, under load (uses docker compose)
    python3 lab.py smoke                  run every demo and check the results (about a minute)
    python3 lab.py show                   guided class run: each demo's chat question, then the demo,
                                          its PASS/FAIL checks and the teaching point. Enter between steps.
    python3 lab.py show --list            the steps.  --only fresh,stale runs some.  --auto runs with no pauses

Where it connects. By default the ports in .env (or the defaults) on localhost.
Set LAB_HOST to drive a lab running on another machine, for example
    LAB_HOST=10.0.0.5 python3 lab.py trace
The drill, smoke and show commands also run docker compose, so run those on the lab machine.
"""
import argparse
import http.client
import json
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Section 2 of the notebook: each hop's own p99 target, in ms (assumed numbers for the hypothetical app).
BUDGET = [("parse_and_arm", "network and request parsing", 10), ("user_features", "fetch user features", 15),
          ("candidates", "cheap search, in parallel", 30), ("item_features", "fetch item features", 20),
          ("light_model", "light model", 15), ("ranking_model", "full ranking model", 60),
          ("value_and_rerank", "combine scores, arrange the page", 10),
          ("log_and_respond", "build the response", 15)]


def env():
    vals = {}
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    vals.update({k: v for k, v in os.environ.items() if k in vals or k.endswith("_PORT") or k == "LAB_HOST"})
    return vals


E = env()
HOST = E.get("LAB_HOST", "127.0.0.1")
FEED = (HOST, int(E.get("FEED_PORT", 8080)))
RETR = (HOST, int(E.get("RETRIEVAL_PORT", 8081)))
GRAFANA = f"http://{HOST if HOST != '127.0.0.1' else 'localhost'}:{E.get('GRAFANA_PORT', 3000)}"


def call(addr, method, path, body=None, timeout=10):
    c = http.client.HTTPConnection(*addr, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    out = r.read()
    c.close()
    if r.status >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status} {out[:200]!r}")
    return json.loads(out)


def compose(*args):
    print(f"$ docker compose {' '.join(args)}", flush=True)
    subprocess.run(["docker", "compose", *args], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


# ---------------------------------------------------------------------------------------------- status
def cmd_status(a):
    ok = True
    for name, addr, path in [("feed", FEED, "/health"), ("retrieval", RETR, "/config")]:
        try:
            r = call(addr, "GET", path, timeout=3)
            extra = f"  {r['items']:,} items, {r['nlist']:,} clusters, nprobe {r['nprobe']}" if name == "retrieval" else ""
            print(f"  {name:10} up   http://{addr[0]}:{addr[1]}{extra}")
        except Exception as e:
            ok = False
            print(f"  {name:10} DOWN ({e.__class__.__name__}). Is the stack up?  docker compose up -d")
    try:
        s = call(FEED, "GET", "/debug/stream", timeout=3)
        print(f"  stream     {s['produced']} events sent, {s['processed']} turned into features")
    except Exception:
        pass
    print(f"  dashboard  {GRAFANA}  (Feed lab)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------- trace
def cmd_trace(a):
    rng = random.Random(1)
    for _ in range(30):                                   # warm caches and connections first
        call(FEED, "GET", f"/feed?user={rng.randrange(1000, 90000)}")
    runs = [call(FEED, "GET", f"/feed?user={rng.randrange(1000, 90000)}&debug=1") for _ in range(a.n)]
    call(FEED, "POST", f"/reset?user={a.user}")
    one = call(FEED, "GET", f"/feed?user={a.user}&debug=1")
    print(f"One feed request for user {one['user']}  (arm {one['arm']}, served by {one['served_by']})\n")
    print(f"{'hop':34}{'this request':>13}{'p99 of ' + str(a.n):>12}{'budget':>8}")
    for key, label, budget in BUDGET:
        p99 = pct([r["hops_ms"][key] for r in runs], 99)
        print(f"{label:34}{one['hops_ms'][key]:>10.1f} ms{p99:>9.1f} ms{budget:>5} ms")
    tot = [r["server_ms"] for r in runs]
    print(f"{'total, server side':34}{one['server_ms']:>10.1f} ms{pct(tot, 99):>9.1f} ms{sum(b for *_, b in BUDGET):>5} ms")
    c = one["candidates"]
    srcs = ", ".join(f"{k} {v:,}" for k, v in c["by_source"].items())
    print(f"\ncandidates  {srcs}")
    print(f"            {c['after_merge_and_seen']:,} after merging and removing seen items, "
          f"{c['after_light_model']} after the light model, 20 on the page")
    print(f"\n{'pos':>3} {'item':>8} {'topic':>5}  {'source':11} {'P(click)':>8} {'P(finish)':>9} {'P(hide)':>8} {'value':>7}")
    for r in one["page"][:a.rows]:
        flag = "  <- exploration slot" if r["explore"] else ""
        f = lambda k, w: f"{r[k]:>{w}.3f}" if k in r else f"{'-':>{w}}"     # explored items skip the ranker
        val = f"{r['value']:>7.3f}" if not r["explore"] else f"{'-':>7}"
        print(f"{r['pos']:>3} {r['item']:>8} {r['topic']:>5}  {r['source']:11} {f('click', 8)} "
              f"{f('finish', 9)} {f('hide', 8)} {val}{flag}")
    topics = Counter(r["topic"] for r in one["page"])
    print(f"\npage of 20: {len(topics)} topics, sources {dict(Counter(r['source'] for r in one['page']))}")
    return {"one": one, "p99_total": pct(tot, 99), "hop_p99": {k: pct([r["hops_ms"][k] for r in runs], 99)
                                                                for k, *_ in BUDGET}}


# ---------------------------------------------------------------------------------------------- sweep
def cmd_sweep(a):
    r = call(RETR, "GET", f"/sweep?users={a.users}&nprobes={a.nprobes}", timeout=600)
    print(f"{r['items']:,} items in {r['nlist']:,} clusters, {r['users']} users, recall@{r['k']} against exact search\n")
    print(f"{'nprobe':>7}{'recall@100':>12}{'vectors scored':>16}{'share':>8}{'mean ms':>9}{'p99 ms':>8}")
    for row in r["rows"]:
        print(f"{row['nprobe']:>7}{row['recall']:>12.3f}{row['scored']:>16,}{row['scored'] / r['items']:>8.1%}"
              f"{row['mean_ms']:>9.2f}{row['p99_ms']:>8.2f}")
    return r["rows"]


def cmd_nprobe(a):
    r = call(RETR, "POST", "/config", {"nprobe": a.value})
    print(f"retrieval now visits {r['nprobe']} clusters per search")
    return 0


# ---------------------------------------------------------------------------------------------- load
def run_load(rps, seconds, label="", quiet=False, stop=None):
    """Open-loop load: requests are sent on a fixed schedule, and latency counts from the scheduled
    time, so a slow server cannot hide its slowness by slowing the sender down."""
    jobs, results, lock = queue.Queue(), [], threading.Lock()
    rng = random.Random(7)

    def worker():
        conn = None
        while True:
            job = jobs.get()
            if job is None:
                return
            t_sched, user = job
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(*FEED, timeout=5)
                conn.request("GET", f"/feed?user={user}")
                resp = conn.getresponse()
                body = json.loads(resp.read())
                rec = (t_sched, time.perf_counter() - t_sched, body.get("served_by", "?") if resp.status == 200 else "error")
            except Exception:
                conn = None
                rec = (t_sched, time.perf_counter() - t_sched, "error")
            with lock:
                results.append(rec)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(48)]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    n, next_report, reported = 0, t0 + 2, 0
    lines = []
    try:
        while time.perf_counter() - t0 < seconds and not (stop and stop.is_set()):
            t_sched = t0 + n / rps
            delay = t_sched - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            jobs.put((t_sched, rng.randrange(0, 90000)))
            n += 1
            if time.perf_counter() >= next_report:
                with lock:
                    window = [r for r in results[reported:]]
                    reported = len(results)
                if window and not quiet:
                    lines.append(report_line(time.perf_counter() - t0, window, label))
                    print(lines[-1], flush=True)
                next_report += 2
    except KeyboardInterrupt:
        pass
    for _ in threads:
        jobs.put(None)
    for t in threads:
        t.join(timeout=6)
    return results


def report_line(t, window, label=""):
    lat = [r[1] * 1e3 for r in window]
    served = Counter(r[2] for r in window)
    fb = sum(v for k, v in served.items() if k not in ("full",)) / len(window)
    return (f"{label}{t:5.0f}s  {len(window) / 2:5.1f} req/s   p50 {pct(lat, 50):6.1f} ms   p99 {pct(lat, 99):6.1f} ms"
            f"   not full ranking {fb:5.1%}   errors {served.get('error', 0)}")


def summary(results, label):
    lat = [r[1] * 1e3 for r in results]
    served = Counter(r[2] for r in results)
    return {"phase": label, "requests": len(results), "p50": pct(lat, 50), "p99": pct(lat, 99),
            "served_by": dict(served)}


def cmd_load(a):
    print(f"Sending {a.rps} feed requests a second for {a.seconds} s. Ctrl-C to stop.")
    print("Latency is measured by the client, from when each request was due to be sent.")
    res = run_load(a.rps, a.seconds)
    s = summary(res, "all")
    print(f"\n{s['requests']} requests, p50 {s['p50']:.1f} ms, p99 {s['p99']:.1f} ms, served by {s['served_by']}")
    return 0


# ---------------------------------------------------------------------------------------------- fresh
def page_topics(user):
    r = call(FEED, "GET", f"/feed?user={user}&debug=1")
    return r, Counter(x["topic"] for x in r["page"])


def show_topics(title, c):
    print(f"{title:28}" + "  ".join(f"topic {t}: {n}" for t, n in c.most_common(5)))


def wait_stream(timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = call(FEED, "GET", "/debug/stream")
        if s["processed"] >= s["produced"]:
            return s, time.time() - t0
        time.sleep(0.05)
    return call(FEED, "GET", "/debug/stream"), None


def cmd_fresh(a):
    call(FEED, "POST", f"/reset?user={a.user}")
    first, before = page_topics(a.user)
    show_topics(f"page 1 for user {a.user}", before)
    hide_topic = before.most_common(1)[0][0]
    victims = [x["item"] for x in first["page"] if x["topic"] == hide_topic][:a.hides]
    print(f"\nThe user hides {len(victims)} items from topic {hide_topic}. Each hide goes app -> Kafka -> "
          "stream worker -> Redis.")
    for item in victims:
        call(FEED, "POST", "/event", {"user": a.user, "item": item, "action": "hide"})
    s, waited = wait_stream(a.wait)
    if waited is None:
        print(f"  NOT PROCESSED after {a.wait:.0f} s: {s['produced'] - s['processed']} events waiting in Kafka. "
              "Is the stream worker running?")
    else:
        print(f"  all events turned into features after {waited * 1e3:.0f} ms "
              f"(the last one took {s['last_event_to_feature_ms']:.0f} ms from app to Redis)")
    second, after = page_topics(a.user)
    show_topics("page 2, after the hides", after)
    print(f"\ntopic {hide_topic} on the page: {before[hide_topic]} before, {after[hide_topic]} after")
    print(f"session features the ranker read for page 2: {second['session']}")
    return {"topic": hide_topic, "before": before[hide_topic], "after": after[hide_topic], "processed": waited is not None,
            "waited_ms": waited * 1e3 if waited is not None else None}


def cmd_page(a):
    """The user's next page, without resetting anything."""
    r, c = page_topics(a.user)
    show_topics(f"next page for user {a.user}", c)
    print(f"session features the ranker read: {r['session']}")
    return 0


# ---------------------------------------------------------------------------------------------- drill
def cmd_drill(a):
    """Healthy, then ranker stopped, then ranker hung, each under the same load."""
    phases = []

    def phase(label, seconds):
        print(f"\n--- {label} ---", flush=True)
        res = run_load(a.rps, seconds, label="  ")
        phases.append(summary(res[len(res) // 5:], label))      # drop the first fifth, while the change settles

    try:
        phase("ranker healthy", a.seconds)
        compose("stop", "ranker")
        phase("ranker stopped", a.seconds)
        compose("start", "ranker")
        time.sleep(4)
        compose("pause", "ranker")
        phase("ranker hung (paused)", a.seconds)
    finally:
        subprocess.run(["docker", "compose", "unpause", "ranker"], cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "compose", "start", "ranker"], cwd=HERE, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    print(f"\n{'phase':24}{'requests':>9}{'p50 ms':>9}{'p99 ms':>9}   served by")
    for p in phases:
        print(f"{p['phase']:24}{p['requests']:>9}{p['p50']:>9.1f}{p['p99']:>9.1f}   {p['served_by']}")
    return phases


# ---------------------------------------------------------------------------------------------- show
def hops_p99(user_seed, n, key):
    """p99 of one hop over n debug requests, for different users."""
    rng = random.Random(user_seed)
    return pct([call(FEED, "GET", f"/feed?user={rng.randrange(1000, 90000)}&debug=1")["hops_ms"][key]
                for _ in range(n)], 99)


def show_trace(a, check):
    t = cmd_trace(argparse.Namespace(user=42, n=200, rows=a.rows))
    check(t["one"]["served_by"] == "full" and len(t["one"]["page"]) == 20,
          "one request is served by the full ranker, a page of 20")
    over = [label for key, label, b in BUDGET if t["hop_p99"][key] > b]
    check(not over, "every hop's p99 is inside its budget" + (f" (over: {', '.join(over)})" if over else ""))
    check(t["p99_total"] < sum(b for *_, b in BUDGET),
          f"the whole request's p99 is {t['p99_total']:.1f} ms, inside 175 ms")
    slow = sorted(BUDGET, key=lambda h: -t["hop_p99"][h[0]])[:2]
    print(f"  read  slowest hops by p99 in this run: {slow[0][1]} {t['hop_p99'][slow[0][0]]:.1f} ms, "
          f"then {slow[1][1]} {t['hop_p99'][slow[1][0]]:.1f} ms")
    hop_sum = sum(t["hop_p99"].values())
    print(f"  note  the eight hop p99s add up to {hop_sum:.1f} ms, the request's p99 is {t['p99_total']:.1f} ms. "
          "They need not agree.")


def show_sweep(a, check):
    rows = cmd_sweep(argparse.Namespace(users=200, nprobes="1,4,16,64,128,256,1024"))
    by = {r["nprobe"]: r for r in rows}
    rec = [r["recall"] for r in rows if r["nprobe"] != "exact"]
    check(all(x <= y + 1e-9 for x, y in zip(rec, rec[1:])), "recall rises with nprobe")
    if 128 in by and 256 in by:
        check(by[128]["recall"] >= 0.9, f"nprobe 128 finds {by[128]['recall']:.3f} of the true top 100")
        print(f"  read  nprobe 128 scores {by[128]['scored'] / by['exact']['scored']:.1%} of the catalogue, "
              f"{by['exact']['mean_ms'] / by[128]['mean_ms']:.0f}x faster than exact search on average")
        ratio = by[256]["scored"] / by[128]["scored"]
        check(ratio > 1.3, f"128 -> 256 clusters costs {ratio:.1f}x the vectors for "
                           f"{by[256]['recall'] - by[128]['recall']:+.3f} recall")


def search_totals():
    """Count and total seconds of ANN searches so far, from the histogram behind 'ANN search time'."""
    c = http.client.HTTPConnection(*RETR, timeout=5)
    c.request("GET", "/metrics")
    text = c.getresponse().read().decode()
    c.close()
    vals = {ln.split()[0]: float(ln.split()[1]) for ln in text.splitlines()
            if ln.startswith(("retrieval_search_seconds_count", "retrieval_search_seconds_sum"))}
    return vals["retrieval_search_seconds_count"], vals["retrieval_search_seconds_sum"]


def show_nprobe(a, check):
    start = call(RETR, "GET", "/config")["nprobe"]
    stop = threading.Event()
    bg = threading.Thread(target=run_load, args=(a.rps, 3600), kwargs={"quiet": True, "stop": stop}, daemon=True)
    print(f"Background traffic at {a.rps:g} requests a second, so the index is busy. "
          "About 10 seconds at each setting.")
    bg.start()
    rows = {}
    try:
        time.sleep(3)
        for n in (128, 1024):
            call(RETR, "POST", "/config", {"nprobe": n})
            time.sleep(1)
            c0, s0 = search_totals()
            hop = [call(FEED, "GET", f"/feed?user={u}&debug=1")["hops_ms"]["candidates"]
                   for u in random.Random(3).sample(range(1000, 90000), 100)]
            c1, s1 = search_totals()
            rows[n] = (1e3 * (s1 - s0) / max(c1 - c0, 1), pct(hop, 50), int(c1 - c0))
            print(f"  nprobe {n:>4}   mean ANN search {rows[n][0]:5.2f} ms over {rows[n][2]} searches"
                  f"   candidates hop p50 {rows[n][1]:5.1f} ms")
    finally:
        call(RETR, "POST", "/config", {"nprobe": start})
        stop.set()
        bg.join(timeout=10)
        print(f"  nprobe put back to {start}")
    lo, hi = rows[128][0], rows[1024][0]
    check(hi > 1.5 * lo, f"8x the clusters makes each search {hi / lo:.1f}x slower under load ({lo:.2f} -> {hi:.2f} ms)")
    print("  note  the whole candidates hop moves less, and noisily, because most of it is moving data, as in trace")


def show_fresh(a, check):
    f = cmd_fresh(argparse.Namespace(user=505, hides=3, wait=10))
    check(f["processed"], "all three hides became features in Redis"
          + (f", {f['waited_ms']:.0f} ms after they were sent" if f["processed"] else ""))
    check(f["after"] < f["before"], f"the hidden topic shrinks on the next page ({f['before']} -> {f['after']})")


def show_stale(a, check):
    compose("stop", "stream-worker")
    try:
        f = cmd_fresh(argparse.Namespace(user=101, hides=3, wait=5))
        check(not f["processed"], "with the worker stopped, the hides wait in Kafka")
        check(f["after"] == f["before"], "and the next page did not change, with no error")
        print()
        cmd_status(a)
    finally:
        compose("start", "stream-worker")
    s, waited = wait_stream(20)
    print()
    cmd_status(a)
    check(waited is not None, f"the restarted worker caught up, {s['processed']} of {s['produced']} events")
    print()
    cmd_page(argparse.Namespace(user=101))


def show_drill(a, check):
    ph = cmd_drill(argparse.Namespace(rps=a.rps, seconds=a.seconds))
    h, s, g = ph
    check(h["served_by"].get("full", 0) > 0.95 * h["requests"], "healthy: the full ranker serves the page")
    check(s["served_by"].get("light_fallback", 0) > 0.9 * s["requests"] and not s["served_by"].get("error"),
          "stopped: the light model's order is served, and no user sees an error")
    check(g["p50"] > 60 and not g["served_by"].get("error"),
          f"hung: still no errors, but p50 jumps from {h['p50']:.0f} ms to {g['p50']:.0f} ms, "
          "since every request now waits out the ranker's deadline")


# Each step: (name, notebook section, title, what to put on screen, chat prompt, function, teaching point, minutes)
SHOW = [
    ("trace", "9.2", "One request, hop by hop",
     "The section 2 latency budget on the whiteboard.",
     "Which hop will be slowest?",
     show_trace,
     "Compare the slowest hops above with the chat's guesses. Candidate search is mostly moving about 1,930 "
     "candidates between services, not the index search, so moving data can cost more than finding it. "
     "Per-hop p99s do not add up to the request's p99, so measure the whole request too.", 4),
    ("sweep", "9.3", "Recall against work, on a real index",
     "The cluster drawing from section 5.",
     "To find 95 of the true top 100, what share of the catalogue must one search score? 1%, 4%, 20% or 50%?",
     show_sweep,
     "Scoring about 4% of the catalogue finds about 95 of the true top 100. Each extra point of recall "
     "costs more than the one before.", 4),
    ("nprobe", "9.3", "The same trade-off under load (optional)",
     "Grafana: the nprobe tile, 'ANN search time' and 'p99 per hop'.",
     "Retrieval will search 8 times as many clusters. Is each search about as fast, a few times slower, or 8 times slower?",
     show_nprobe,
     "A few times slower, not 8x, since each search has a fixed cost. More recall is still paid for in latency "
     "on every request, and at 100x the catalogue that cost no longer fits in 30 ms, which is Checkpoint 9.1.", 3),
    ("fresh", "9.4", "A feed that reacts",
     "The lab picture in 9.1, to trace app -> Kafka -> stream worker -> Redis.",
     "User 505's first page has 16 items from one topic. After three hides, how many from that topic stay "
     "on the next page?",
     show_fresh,
     "The hides became features within a second, and the very next page changed. The user tower moved, so "
     "retrieval found different items, and the ranker read the hides as a feature.", 3),
    ("stale", "9.4", "Deliberate failure: the stream stops",
     "Grafana's 'Stream backlog' tile.",
     "The stream worker is stopped and the user hides three items. Did the user see an error?",
     show_stale,
     "No error, a full page, quickly. Just stale. A stale feed looks healthy. Only the backlog shows it, "
     "and Kafka kept the events, so the worker caught up. Alert on the backlog.", 4),
    ("drill", "9.5", "When the ranker fails, then hangs",
     "Grafana's 'How requests were served' panel and the latency panel.",
     "The ranker is stopped under load. Did any user see an error? Then, when it hangs instead, what happens "
     "to p50?",
     show_drill,
     "Dead is cheap, the fallback serves at once. Hung is expensive, every request waits the full deadline. "
     "Set deadlines from the budget and alert on the fallback share.", 5),
]


def cmd_show(a):
    """Guided walk through every demo in section 9, in class order."""
    if a.list:
        print(f"{'step':8}{'section':>8}  {'minutes':>7}  title")
        for name, sec, title, *_, mins in SHOW:
            print(f"{name:8}{sec:>8}  {mins:>7}  {title}")
        return 0
    steps = [s for s in SHOW if not a.only or s[0] in a.only.split(",")]
    unknown = set(a.only.split(",")) - {s[0] for s in SHOW} if a.only else set()
    if unknown:
        print(f"unknown step(s): {', '.join(sorted(unknown))}. See: python3 lab.py show --list")
        return 2
    auto = a.auto or not sys.stdin.isatty()
    results = []
    bar = "=" * 78
    try:
        for i, (name, sec, title, screen, prompt, fn, point, mins) in enumerate(steps, 1):
            print(f"\n{bar}\nSTEP {i} of {len(steps)}   {title}   (notebook {sec}, about {mins} min)\n{bar}")
            print(f"ON SCREEN       {screen}")
            print(f"POST IN CHAT    \"{prompt}\"")
            if not auto:
                ans = input("\nEnter to run it, s to skip, q to stop the show: ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    results.append((name, "skipped", []))
                    continue
            print()
            fails = []

            def check(cond, msg):
                print(f"  {'PASS' if cond else 'FAIL'}  {msg}", flush=True)
                if not cond:
                    fails.append(msg)

            try:
                fn(a, check)
            except Exception as e:                       # one broken demo should not end the class
                check(False, f"the demo raised {e.__class__.__name__}: {e}")
            print(f"\nTEACHING POINT  {point}")
            results.append((name, "passed" if not fails else "FAILED", fails))
            if not auto and i < len(steps):
                input("\nEnter for the next demo: ")
    except (KeyboardInterrupt, EOFError):
        print("\nshow stopped")
    finally:                                             # leave the lab as a class expects to find it
        for args in (("unpause", "ranker"), ("start", "ranker", "stream-worker")):
            subprocess.run(["docker", "compose", *args], cwd=HERE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    print(f"\n{bar}\nSHOW SUMMARY")
    for name, status, fails in results:
        print(f"  {name:8}{status}" + (f"   {'; '.join(fails)}" if fails else ""))
    bad = [r for r in results if r[1] == "FAILED"]
    print("ALL DEMOS PASSED" if results and not bad else f"{len(bad)} demo(s) failed. Use the recorded runs in "
          "the notebook for those.")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------- smoke
def cmd_smoke(a):
    failures = []

    def check(cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}", flush=True)
        if not cond:
            failures.append(msg)

    print("== status")
    check(cmd_status(a) == 0, "feed and retrieval answer")
    print("\n== trace")
    r = call(FEED, "GET", "/feed?user=11&debug=1")
    check(r["served_by"] == "full" and len(r["page"]) == 20, "a full page of 20, served by the full ranker")
    check(set(r["hops_ms"]) == {k for k, *_ in BUDGET}, "every hop timed")
    check(any(x["explore"] for x in r["page"]), "one exploration slot on the page")
    print("\n== sweep")
    s = call(RETR, "GET", "/sweep?users=100&nprobes=1,8,64,512", timeout=600)["rows"]
    rec = [row["recall"] for row in s[1:]]
    check(all(x <= y + 1e-9 for x, y in zip(rec, rec[1:])), f"recall rises with nprobe {[round(x, 3) for x in rec]}")
    check(s[-1]["mean_ms"] > s[1]["mean_ms"], "search time rises with nprobe")
    print("\n== fresh, stream running")
    a.user, a.hides, a.wait = 505, 3, 10
    f = cmd_fresh(a)
    check(f["processed"] and f["after"] < f["before"], "hides reach the ranker and the hidden topic shrinks")
    print("\n== fresh, stream worker stopped")
    compose("stop", "stream-worker")
    a.user, a.wait = 101, 3
    f = cmd_fresh(a)
    check(not f["processed"], "with the worker stopped, events wait in Kafka")
    compose("start", "stream-worker")
    s, waited = wait_stream(20)
    check(waited is not None, "the restarted worker catches up on the backlog")
    print("\n== drill")
    a.rps, a.seconds = a.rps or 15, 12
    ph = cmd_drill(a)
    check(ph[0]["served_by"].get("full", 0) > 0.95 * ph[0]["requests"], "healthy: served by the full ranker")
    check(ph[1]["served_by"].get("light_fallback", 0) > 0.9 * ph[1]["requests"], "stopped: light-model fallback")
    check(ph[1]["served_by"].get("error", 0) == 0, "stopped: no errors reach the user")
    check(ph[2]["p99"] > ph[0]["p99"], "hung: p99 rises toward the ranker deadline")
    print(f"\n{'SMOKE PASSED' if not failures else 'SMOKE FAILED: ' + '; '.join(failures)}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("trace")
    p.add_argument("--user", type=int, default=42)
    p.add_argument("--n", type=int, default=200, help="requests behind the p99 column")
    p.add_argument("--rows", type=int, default=20)
    p = sub.add_parser("sweep")
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--nprobes", default="1,4,16,64,128,256,1024")
    p = sub.add_parser("nprobe")
    p.add_argument("value", type=int)
    p = sub.add_parser("load")
    p.add_argument("--rps", type=float, default=20)
    p.add_argument("--seconds", type=float, default=600)
    p = sub.add_parser("fresh")
    p.add_argument("--user", type=int, default=505)
    p.add_argument("--hides", type=int, default=3)
    p.add_argument("--wait", type=float, default=5)
    p = sub.add_parser("page")
    p.add_argument("--user", type=int, default=505)
    p = sub.add_parser("drill")
    p.add_argument("--rps", type=float, default=20)
    p.add_argument("--seconds", type=float, default=20)
    p = sub.add_parser("smoke")
    p.add_argument("--rps", type=float, default=15)
    p = sub.add_parser("show", help="guided run of every demo in class order")
    p.add_argument("--auto", action="store_true", help="no pauses, for a rehearsal or a recording")
    p.add_argument("--only", default="", help="comma-separated steps, for example fresh,stale")
    p.add_argument("--list", action="store_true", help="list the steps and stop")
    p.add_argument("--rows", type=int, default=5, help="page rows the trace step prints")
    p.add_argument("--rps", type=float, default=20)
    p.add_argument("--seconds", type=float, default=20, help="length of each drill phase")
    a = ap.parse_args()
    out = globals()[f"cmd_{a.cmd}"](a)
    sys.exit(out if isinstance(out, int) else 0)


if __name__ == "__main__":
    main()
