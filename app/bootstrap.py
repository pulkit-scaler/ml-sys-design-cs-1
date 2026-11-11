"""One-off job: build the synthetic world, the ANN index and the rankers, and load Redis.

Runs once per data volume. A second run sees the marker files and only reloads
Redis if Redis was emptied.
"""
import json
import os
import time

import faiss
import joblib
import numpy as np
import redis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import world as W

R = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379)
CONFIG = f"items={W.N_ITEMS} users={W.N_USERS} seed={W.SEED} v7"


def log(msg):
    print(f"[bootstrap {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_files():
    t0 = time.time()
    w = W.make_world()
    log(f"world made: {W.N_ITEMS:,} items, {W.N_USERS:,} users, {W.N_TOPICS} topics ({time.time() - t0:.1f}s)")
    np.save(f"{W.DATA}/vecs.npy", w["vecs"])
    np.savez(f"{W.DATA}/world.npz", **{k: v for k, v in w.items() if k != "vecs"})

    # The ANN index. Faiss suggests 4 to 16 x sqrt(N) clusters; 4 x sqrt(1M) = 4,000.
    nlist = int(4 * np.sqrt(W.N_ITEMS))
    t0 = time.time()
    quant = faiss.IndexFlatIP(W.DIM)
    index = faiss.IndexIVFFlat(quant, W.DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(W.SEED)
    sample = w["vecs"][rng.choice(W.N_ITEMS, min(W.N_ITEMS, 60 * nlist), replace=False)]
    index.train(sample)
    index.add(w["vecs"])
    faiss.write_index(index, f"{W.DATA}/ivf.index")
    log(f"IVF index built: {nlist:,} clusters ({time.time() - t0:.1f}s)")

    # The trending list: most-watched items uploaded in the last 7 days.
    recent = np.flatnonzero(w["age_h"] < 7 * 24)
    trending = recent[np.argsort(-w["log_pop"][recent])[:500]].astype(np.int32)
    np.save(f"{W.DATA}/trending.npy", trending)

    train_rankers(w)


def training_log(w, n, rng):
    """Impressions as the previous system logged them, with features and labels."""
    users = rng.integers(W.N_USERS, size=n)
    taste = w["taste"][users]
    # Most impressions came from the user's own topics, some from anywhere.
    own = rng.random(n) < 0.6
    topic_own = np.array([rng.choice(W.N_TOPICS, p=t) for t in taste[own]])
    by_topic = [np.flatnonzero(w["topic"] == t) for t in range(W.N_TOPICS)]
    items = rng.integers(W.N_ITEMS, size=n)
    items[own] = np.array([rng.choice(by_topic[t]) for t in topic_own])
    topic = w["topic"][items]
    # A session state per impression: engaged and hidden counts on a few topics.
    sess_eng = np.zeros((n, W.N_TOPICS), np.float32)
    sess_hid = np.zeros((n, W.N_TOPICS), np.float32)
    active = rng.random(n) < 0.6
    for i in np.flatnonzero(active):
        sess_eng[i, rng.choice(W.N_TOPICS, p=taste[i])] += rng.poisson(2)
        if rng.random() < 0.4:
            sess_hid[i, rng.choice(W.N_TOPICS, p=taste[i])] += rng.integers(1, 6)
    uvec = W.user_tower_batch(taste, sess_eng, sess_hid, w["personal"][users], w["centres"])
    dot = np.einsum("ij,ij->i", uvec, np.load(f"{W.DATA}/vecs.npy", mmap_mode="r")[items])
    rows = np.arange(n)
    taste_aff = taste[rows, topic] * W.N_TOPICS
    se, sh = sess_eng[rows, topic], sess_hid[rows, topic]
    X = np.column_stack([dot, taste_aff, se, sh, w["item_ctr"][items], w["bait_score"][items],
                         w["log_age_h"][items], w["log_pop"][items]]).astype(np.float32)
    probs = W.true_probs(taste_aff, se, sh, w["quality"][items], w["bait"][items])
    Y = np.column_stack([rng.random(n) < p for p in probs]).astype(np.int8)
    return X, Y


def train_rankers(w):
    rng = np.random.default_rng(W.SEED + 1)
    t0 = time.time()
    X, Y = training_log(w, 240_000, rng)
    log(f"training log: {len(X):,} impressions, label rates "
        + ", ".join(f"{h} {Y[:, j].mean():.3f}" for j, h in enumerate(W.HEADS)) + f" ({time.time() - t0:.1f}s)")
    split = int(0.85 * len(X))
    t0 = time.time()
    heavy = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40,
                                                           early_stopping=True, random_state=0))
    heavy.fit(X[:split], Y[:split])
    p = heavy.predict_proba(X[split:])
    aucs = {h: round(float(roc_auc_score(Y[split:, j], p[:, j])), 3) for j, h in enumerate(W.HEADS)}
    log(f"multi-task ranker trained, holdout AUC {aucs} ({time.time() - t0:.1f}s)")
    li = [W.FEATURES.index(f) for f in W.LIGHT_FEATURES]
    light = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    light.fit(X[:split][:, li], Y[:split, 0])
    lauc = roc_auc_score(Y[split:, 0], light.predict_proba(X[split:][:, li])[:, 1])
    log(f"light ranker trained, holdout click AUC {lauc:.3f}")
    joblib.dump(heavy, f"{W.DATA}/heavy.joblib")
    joblib.dump(light, f"{W.DATA}/light.joblib")
    json.dump({"heavy_auc": aucs, "light_click_auc": round(float(lauc), 3)}, open(f"{W.DATA}/models.json", "w"))


def load_redis():
    w = np.load(f"{W.DATA}/world.npz")
    t0 = time.time()
    R.flushdb()
    items = np.column_stack([w[f] for f in W.ITEM_FIELDS]).astype(np.float32)
    users = np.hstack([w["taste"], w["personal"]]).astype(np.float32)
    for start in range(0, W.N_ITEMS, 50_000):
        R.mset({f"i:{i}": items[i].tobytes() for i in range(start, min(start + 50_000, W.N_ITEMS))})
    for start in range(0, W.N_USERS, 50_000):
        R.mset({f"u:{u}": users[u].tobytes() for u in range(start, min(start + 50_000, W.N_USERS))})
    R.set("trending", np.load(f"{W.DATA}/trending.npy").tobytes())
    R.set("seeded", CONFIG)
    log(f"redis loaded: {W.N_ITEMS:,} item records, {W.N_USERS:,} user records ({time.time() - t0:.1f}s)")


def create_topics():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": os.environ.get("KAFKA", "redpanda:9092")})
    wanted = [NewTopic("events", num_partitions=3, replication_factor=1),
              NewTopic("impressions", num_partitions=3, replication_factor=1)]
    for name, fut in admin.create_topics(wanted).items():
        try:
            fut.result()
            log(f"topic '{name}' created")
        except Exception as e:                     # already exists on a second run
            log(f"topic '{name}' already exists" if "ALREADY_EXISTS" in str(e) else f"topic '{name}': {e}")


def main():
    os.makedirs(W.DATA, exist_ok=True)
    create_topics()
    marker = f"{W.DATA}/built.txt"
    if not (os.path.exists(marker) and open(marker).read() == CONFIG):
        build_files()
        open(marker, "w").write(CONFIG)
    else:
        log("data volume already built, skipping")
    if (R.get("seeded") or b"").decode() != CONFIG:
        load_redis()
    else:
        log("redis already loaded, skipping")
    log("done")


if __name__ == "__main__":
    main()
