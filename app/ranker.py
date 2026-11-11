"""Multi-task ranker service: one small network with a shared body and four heads.

Takes a feature matrix for up to a few hundred candidates and returns
P(click), P(finish), P(like) and P(hide) for each.
"""
import os
import time

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel

import world as W

MODEL = joblib.load(f"{W.DATA}/heavy.joblib")
EXTRA_MS = float(os.environ.get("RANKER_EXTRA_MS", 0))   # optional: pretend the model is bigger
app = FastAPI()
SCORE = Histogram("ranker_score_seconds", "Time to score one request's candidates",
                  buckets=[.001, .002, .004, .008, .016, .032, .064, .128, .256])


class Req(BaseModel):
    features: list[list[float]]


@app.post("/score")
def score(req: Req):
    t = time.perf_counter()
    X = np.asarray(req.features, dtype=np.float32)
    p = MODEL.predict_proba(X)
    if EXTRA_MS:
        time.sleep(EXTRA_MS / 1e3)
    dt = time.perf_counter() - t
    SCORE.observe(dt)
    return {"heads": W.HEADS, "p": p.round(5).tolist(), "ms": dt * 1e3}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
