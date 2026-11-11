"""Candidate generation service: several sources over a 1M-item catalogue.

  two_tower       approximate nearest-neighbour search on a Faiss IVF index
  more_like_last  the same index, searched from the last item the user finished
  new_items       recent uploads, searched exactly on their own (the main index is rebuilt rarely)
  trending        the popular list, from memory
"""
import os
import threading
import time

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel

import world as W

faiss.omp_set_num_threads(1)                      # one core per query, so concurrent requests share the CPUs
VECS = np.load(f"{W.DATA}/vecs.npy")
WORLD = np.load(f"{W.DATA}/world.npz")
INDEX = faiss.read_index(f"{W.DATA}/ivf.index")
TRENDING = np.load(f"{W.DATA}/trending.npy")
NEW_IDS = np.flatnonzero(WORLD["age_h"] < 24).astype(np.int64)   # uploaded in the last day
NEW_VECS = VECS[NEW_IDS]
CONF = {"nprobe": int(os.environ.get("NPROBE", 128)), "k": 1500}
LOCK = threading.Lock()

app = FastAPI()
SEARCH = Histogram("retrieval_search_seconds", "ANN search time for the two-tower source",
                   buckets=[.0005, .001, .002, .004, .008, .016, .032, .064, .128, .256])
NPROBE = Gauge("retrieval_nprobe", "IVF clusters visited per search")
NPROBE.set(CONF["nprobe"])


class Req(BaseModel):
    user_vec: list[float]
    last_item: int | None = None


def ann(q, k, nprobe):
    params = faiss.SearchParametersIVF(nprobe=nprobe)
    _, ids = INDEX.search(q.reshape(1, -1), k, params=params)
    return ids[0][ids[0] >= 0]


@app.post("/candidates")
def candidates(req: Req):
    q = np.asarray(req.user_vec, dtype=np.float32)
    nprobe, k = CONF["nprobe"], CONF["k"]
    ms, out = {}, []

    t = time.perf_counter()
    ids = ann(q, k, nprobe)
    dt = time.perf_counter() - t
    SEARCH.observe(dt)
    ms["two_tower"] = dt * 1e3
    out.append(("two_tower", ids))

    t = time.perf_counter()
    if req.last_item is not None:
        out.append(("more_like_last", ann(VECS[req.last_item], 100, nprobe)))
    ms["more_like_last"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    s = np.dot(NEW_VECS, q)
    out.append(("new_items", NEW_IDS[np.argpartition(-s, 150)[:150]]))
    ms["new_items"] = (time.perf_counter() - t) * 1e3

    out.append(("trending", TRENDING[:300]))

    # Merge: first source wins a duplicate.
    seen, ids_all, src_all = set(), [], []
    for name, arr in out:
        for i in arr.tolist():
            if i not in seen:
                seen.add(i)
                ids_all.append(i)
                src_all.append(name)
    ids_np = np.asarray(ids_all, dtype=np.int64)
    dots = np.dot(VECS[ids_np], q)
    return {"ids": ids_all, "source": src_all, "dot": dots.round(5).tolist(), "ms": ms,
            "nprobe": nprobe, "counts": {name: int(len(arr)) for name, arr in out}}


class Conf(BaseModel):
    nprobe: int


@app.post("/config")
def set_config(c: Conf):
    CONF["nprobe"] = max(1, min(c.nprobe, INDEX.nlist))
    NPROBE.set(CONF["nprobe"])
    return CONF


@app.get("/config")
def get_config():
    return {**CONF, "nlist": INDEX.nlist, "items": int(INDEX.ntotal)}


@app.get("/sweep")
def sweep(nprobes: str = "1,2,4,8,16,32,64,128", users: int = 200, k: int = 100):
    """Recall@k against exact search, and the time each search takes, for several nprobe values."""
    with LOCK:                                    # one sweep at a time
        rng = np.random.default_rng(3)
        who = rng.choice(len(WORLD["taste"]), users, replace=False)
        taste = WORLD["taste"][who]
        z = np.zeros_like(taste)
        Q = W.user_tower_batch(taste, z, z, WORLD["personal"][who], WORLD["centres"])
        exact, t_exact = [], []
        for q in Q:
            t = time.perf_counter()
            s = np.dot(VECS, q)
            exact.append(set(np.argpartition(-s, k)[:k].tolist()))
            t_exact.append(time.perf_counter() - t)
        rows = [{"nprobe": "exact", "recall": 1.0, "mean_ms": 1e3 * float(np.mean(t_exact)),
                 "p99_ms": 1e3 * float(np.percentile(t_exact, 99)), "scored": int(INDEX.ntotal)}]
        sizes = np.array([INDEX.invlists.list_size(i) for i in range(INDEX.nlist)])
        for p in [int(x) for x in nprobes.split(",")]:
            _, probed = INDEX.quantizer.search(Q, p)
            scored = INDEX.nlist + sizes[probed].sum(axis=1)
            rec, times = [], []
            for q, truth in zip(Q, exact):
                t = time.perf_counter()
                ids = ann(q, k, p)
                times.append(time.perf_counter() - t)
                rec.append(len(truth.intersection(ids.tolist())) / k)
            rows.append({"nprobe": p, "recall": float(np.mean(rec)), "mean_ms": 1e3 * float(np.mean(times)),
                         "p99_ms": 1e3 * float(np.percentile(times, 99)), "scored": int(scored.mean())})
        return {"items": int(INDEX.ntotal), "nlist": INDEX.nlist, "users": users, "k": k, "rows": rows}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
