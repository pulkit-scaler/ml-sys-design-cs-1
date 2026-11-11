"""Feed service: the request path from the class, hop by hop.

  1 parse and assign A/B arm
  2 fetch user features      (Redis: batch taste + session features from the stream)
  3 candidate sources        (retrieval service)
  4 fetch item features      (Redis)
  5 light model              (in process, 2,000 -> 500)
  6 full ranking model       (ranker service, four heads)
  7 value model and re-rank  (weights, variety rule, one exploration slot)
  8 log and respond          (impressions to Kafka, seen set to Redis)

Fallbacks: if the ranker fails or misses its deadline, serve the light model's
order. If retrieval fails, rank the trending list. Every response says which.
"""
import hashlib
import json
import os
import time

import httpx
import joblib
import numpy as np
import redis.asyncio as aioredis
from confluent_kafka import Producer
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

import world as W

RETRIEVAL = os.environ.get("RETRIEVAL_URL", "http://retrieval:8000")
RANKER = os.environ.get("RANKER_URL", "http://ranker:8000")
RANKER_TIMEOUT = float(os.environ.get("RANKER_TIMEOUT_MS", 80)) / 1e3
RETRIEVAL_TIMEOUT = float(os.environ.get("RETRIEVAL_TIMEOUT_MS", 60)) / 1e3
WEIGHTS = json.loads(os.environ.get("VALUE_WEIGHTS", '{"click": 1, "finish": 4, "like": 10, "hide": -8}'))
VARIETY = float(os.environ.get("VARIETY_PENALTY", 0.05))   # share of the best value lost per topic repeat
PAGE, KEEP = 20, 500
EXPLORE_SLOT = 14                                  # 0-based position of the one exploration slot

WORLD = np.load(f"{W.DATA}/world.npz")
CENTRES = WORLD["centres"]
LIGHT = joblib.load(f"{W.DATA}/light.joblib")
LIGHT_IDX = [W.FEATURES.index(f) for f in W.LIGHT_FEATURES]
TRENDING = np.load(f"{W.DATA}/trending.npy")

R = aioredis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379)
KAFKA = Producer({"bootstrap.servers": os.environ.get("KAFKA", "redpanda:9092"), "linger.ms": 5,
                  "message.timeout.ms": 5000})
HTTP = httpx.AsyncClient(limits=httpx.Limits(max_connections=200, max_keepalive_connections=100))
app = FastAPI()

HOPS = ["parse_and_arm", "user_features", "candidates", "item_features", "light_model",
        "ranking_model", "value_and_rerank", "log_and_respond"]
HOP = Histogram("feed_hop_seconds", "Time spent in each hop", ["hop"],
                buckets=[.0005, .001, .002, .004, .008, .016, .032, .064, .08, .1, .128, .256, .512])
TOTAL = Histogram("feed_request_seconds", "Whole feed request, server side",
                  buckets=[.005, .01, .02, .04, .06, .08, .1, .12, .15, .2, .3, .5, 1])
REQS = Counter("feed_requests_total", "Feed requests by how they were served", ["served_by"])
EVENTS = Counter("feed_events_total", "User actions sent to the event stream", ["action"])
BACKLOG = Gauge("stream_backlog_events", "Actions sent to the stream and not yet turned into features")


def arm_of(user):
    return "B" if int(hashlib.md5(f"exp-variety:{user}".encode()).hexdigest(), 16) % 2 else "A"


async def user_features(user):
    raw, sess, seen = await R.get(f"u:{user}"), await R.hgetall(f"s:{user}"), await R.smembers(f"seen:{user}")
    if raw:                                         # known user: long-term taste and personal vector
        u = np.frombuffer(raw, dtype=np.float32)
        taste, personal = u[:W.N_TOPICS], u[W.N_TOPICS:]
    else:                                           # cold start: no history, average taste
        taste, personal = np.full(W.N_TOPICS, 1 / W.N_TOPICS, np.float32), np.zeros(W.DIM, np.float32)
    eng, hid, last = np.zeros(W.N_TOPICS, np.float32), np.zeros(W.N_TOPICS, np.float32), None
    for k, v in sess.items():
        k = k.decode()
        if k.startswith("e:"):
            eng[int(k[2:])] = float(v)
        elif k.startswith("h:"):
            hid[int(k[2:])] = float(v)
        elif k == "last":
            last = int(v)
    return taste, personal, eng, hid, last, {int(x) for x in seen}, raw is not None


class Event(BaseModel):
    user: int
    item: int
    action: str                                     # click, finish, like or hide


@app.post("/event")
async def event(e: Event):
    if e.action not in W.HEADS:
        raise HTTPException(400, f"action must be one of {W.HEADS}")
    rec = await R.get(f"i:{e.item}")
    topic = int(np.frombuffer(rec, dtype=np.float32)[0]) if rec else -1
    KAFKA.produce("events", json.dumps({"user": e.user, "item": e.item, "topic": topic, "action": e.action,
                                        "ts": time.time()}).encode(), key=str(e.user))
    KAFKA.poll(0)
    await R.incr("stream:produced")
    EVENTS.labels(e.action).inc()
    return {"sent": True, "topic": topic}


@app.get("/feed")
async def feed(user: int, debug: int = 0):
    t_start = time.perf_counter()
    ms, served_by = {}, "full"
    mark = time.perf_counter()

    def lap(hop):
        nonlocal mark
        now = time.perf_counter()
        ms[hop] = (now - mark) * 1e3
        HOP.labels(hop).observe(now - mark)
        mark = now

    arm = arm_of(user)
    lap("parse_and_arm")

    taste, personal, eng, hid, last, seen, known = await user_features(user)
    lap("user_features")

    uvec = W.user_tower(taste, eng, hid, personal, CENTRES)
    try:
        r = await HTTP.post(f"{RETRIEVAL}/candidates", json={"user_vec": uvec.tolist(), "last_item": last},
                            timeout=RETRIEVAL_TIMEOUT)
        r.raise_for_status()
        c = r.json()
        ids, src, dot, src_ms, counts = (np.array(c["ids"]), np.array(c["source"]), np.array(c["dot"]),
                                         c["ms"], c["counts"])
    except Exception:
        served_by = "trending_fallback"
        ids, src, dot = TRENDING.astype(np.int64), np.full(len(TRENDING), "trending"), np.zeros(len(TRENDING))
        src_ms, counts = {}, {"trending": len(TRENDING)}
    keep = np.array([i not in seen for i in ids.tolist()], dtype=bool)
    ids, src, dot = ids[keep], src[keep], dot[keep]
    lap("candidates")

    recs = await R.mget([f"i:{i}" for i in ids.tolist()])
    ok = np.array([x is not None for x in recs])
    ids, src, dot = ids[ok], src[ok], dot[ok]
    items = np.frombuffer(b"".join(x for x in recs if x is not None), dtype=np.float32).reshape(-1, 5)
    topic = items[:, 0].astype(int)
    X = np.column_stack([dot, taste[topic] * W.N_TOPICS, eng[topic], hid[topic],
                         items[:, 1], items[:, 2], items[:, 3], items[:, 4]]).astype(np.float32)
    lap("item_features")

    light = LIGHT.predict_proba(X[:, LIGHT_IDX])[:, 1]
    top = np.argsort(-light)[:KEEP]
    lap("light_model")

    heads = None
    if served_by == "full":
        try:
            r = await HTTP.post(f"{RANKER}/score", json={"features": X[top].tolist()}, timeout=RANKER_TIMEOUT)
            r.raise_for_status()
            heads = np.array(r.json()["p"])
        except Exception:
            served_by = "light_fallback"
    lap("ranking_model")

    if heads is not None:
        value = sum(WEIGHTS[h] * heads[:, j] for j, h in enumerate(W.HEADS))
    else:
        value = light[top]                             # fallback: the light model's order
    # Re-rank: greedy page with a penalty for each repeat of a topic, plus one exploration slot.
    order = top[np.argsort(-value)][:100]
    vmap = dict(zip(top.tolist(), value.tolist()))
    penalty = (VARIETY if arm == "A" else 2 * VARIETY) * max(vmap[order[0]], 1e-9)
    page, pool, used = [], list(order), {}
    while len(page) < PAGE and pool:
        best = max(pool, key=lambda i: vmap[i] - penalty * used.get(topic[i], 0))
        page.append(best)
        pool.remove(best)
        used[topic[best]] = used.get(topic[best], 0) + 1
    explore = None
    new_pool = [i for i in np.flatnonzero(src == "new_items") if i not in page]
    if new_pool and len(page) >= EXPLORE_SLOT:
        explore = max(new_pool, key=lambda i: dot[i])
        page.insert(EXPLORE_SLOT, explore)
        page = page[:PAGE]
    lap("value_and_rerank")

    out = []
    for pos, i in enumerate(page):
        row = {"pos": pos + 1, "item": int(ids[i]), "topic": int(topic[i]), "source": str(src[i]),
               "explore": bool(i == explore), "value": round(float(vmap.get(i, 0.0)), 4)}
        if heads is not None and i in vmap:
            j = int(np.flatnonzero(top == i)[0])
            row.update({h: round(float(heads[j, k]), 4) for k, h in enumerate(W.HEADS)})
        out.append(row)
    try:
        KAFKA.produce("impressions", json.dumps({"ts": time.time(), "user": user, "arm": arm,
                                                 "served_by": served_by, "page": out}).encode(), key=str(user))
        KAFKA.poll(0)
    except BufferError:
        pass
    if page:
        await R.sadd(f"seen:{user}", *[int(ids[i]) for i in page])
        await R.expire(f"seen:{user}", 3600)
    lap("log_and_respond")

    total = time.perf_counter() - t_start
    TOTAL.observe(total)
    REQS.labels(served_by).inc()
    resp = {"user": user, "known_user": known, "arm": arm, "served_by": served_by, "page": out,
            "server_ms": round(total * 1e3, 2)}
    if debug:
        resp.update({"hops_ms": {k: round(v, 2) for k, v in ms.items()},
                     "sources_ms": {k: round(v, 2) for k, v in src_ms.items()},
                     "candidates": {"by_source": counts, "after_merge_and_seen": int(len(ids)),
                                    "after_light_model": int(len(top))},
                     "session": {"engaged": {int(t): float(eng[t]) for t in np.flatnonzero(eng)},
                                 "hidden": {int(t): float(hid[t]) for t in np.flatnonzero(hid)}}})
    return resp


@app.get("/debug/stream")
async def stream_stats():
    produced, processed, lag = await R.mget("stream:produced", "stream:processed", "stream:last_lag_ms")
    return {"produced": int(produced or 0), "processed": int(processed or 0),
            "last_event_to_feature_ms": float(lag or 0)}


@app.post("/reset")
async def reset(user: int):
    """Clear one user's session and seen set, so a demo starts clean."""
    await R.delete(f"s:{user}", f"seen:{user}")
    return {"reset": user}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    produced, processed = await R.mget("stream:produced", "stream:processed")
    BACKLOG.set(int(produced or 0) - int(processed or 0))
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
