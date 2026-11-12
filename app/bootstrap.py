"""One-off job: build the world from KuaiRand-1K, the ANN index and the rankers, and load Redis.

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
CONFIG = f"kuairand-1k items={W.N_ITEMS} seed={W.SEED} v9"


def log(msg):
    print(f"[bootstrap {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_files():
    t0 = time.time()
    w, train = W.make_world()
    n_items, n_users = len(w["topic"]), len(w["taste"])
    log(f"world made: {n_items:,} videos, {n_users:,} users, {W.N_TOPICS} topics ({time.time() - t0:.1f}s)")
    np.save(f"{W.DATA}/vecs.npy", w["vecs"])
    np.savez(f"{W.DATA}/world.npz", **{k: v for k, v in w.items() if k != "vecs"})

    # The ANN index. Faiss suggests 4 to 16 x sqrt(N) clusters; 4 x sqrt(1M) = 4,000.
    nlist = int(4 * np.sqrt(n_items))
    t0 = time.time()
    quant = faiss.IndexFlatIP(W.DIM)
    index = faiss.IndexIVFFlat(quant, W.DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(W.SEED)
    sample = w["vecs"][rng.choice(n_items, min(n_items, 60 * nlist), replace=False)]
    index.train(sample)
    index.add(w["vecs"])
    faiss.write_index(index, f"{W.DATA}/ivf.index")
    log(f"IVF index built: {nlist:,} clusters ({time.time() - t0:.1f}s)")

    # The trending list: most-shown videos uploaded in the last 7 days.
    recent = np.flatnonzero(w["age_h"] < 7 * 24)
    trending = recent[np.argsort(-w["log_pop"][recent])[:500]].astype(np.int32)
    np.save(f"{W.DATA}/trending.npy", trending)

    train_rankers(w, train)


def features(w, train):
    """The ranker's features for each real logged impression, as the feed computes them."""
    users, items = train["user"], train["item"]
    rows = np.arange(len(items))
    topic = w["topic"][items]
    taste = w["taste"][users]
    uvec = W.user_tower_batch(taste, train["eng"], train["hid"], w["personal"][users], w["centres"])
    dot = np.einsum("ij,ij->i", uvec, w["vecs"][items])
    return np.column_stack([dot, taste[rows, topic] * W.N_TOPICS, train["eng"][rows, topic], train["hid"][rows, topic],
                            w["item_ctr"][items], w["skip_rate"][items], w["log_age_h"][items],
                            w["log_pop"][items]]).astype(np.float32)


def train_rankers(w, train):
    t0 = time.time()
    X, Y = features(w, train), train["labels"]
    log(f"training log: {len(X):,} real impressions, label rates "
        + ", ".join(f"{h} {Y[:, j].mean():.4f}" for j, h in enumerate(W.HEADS)) + f" ({time.time() - t0:.1f}s)")
    split = int(0.85 * len(X))
    t0 = time.time()
    heavy = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40,
                                                           early_stopping=True, random_state=0))
    heavy.fit(X[:split], Y[:split])
    p = heavy.predict_proba(X[split:])
    aucs = {h: round(float(roc_auc_score(Y[split:, j], p[:, j])), 3) for j, h in enumerate(W.HEADS)}
    log(f"multi-task ranker trained on {split:,} impressions, checked on {len(X) - split:,} held out, "
        f"holdout AUC {aucs} ({time.time() - t0:.1f}s)")
    log("positive labels, trained on / held out: " + ", ".join(
        f"{h} {int(Y[:split, j].sum()):,} / {int(Y[split:, j].sum()):,}" for j, h in enumerate(W.HEADS)))
    li = [W.FEATURES.index(f) for f in W.LIGHT_FEATURES]
    light = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    light.fit(X[:split][:, li], Y[:split, 0])
    lauc = roc_auc_score(Y[split:, 0], light.predict_proba(X[split:][:, li])[:, 1])
    log(f"light ranker trained, holdout click AUC {lauc:.3f}")
    joblib.dump(heavy, f"{W.DATA}/heavy.joblib")
    joblib.dump(light, f"{W.DATA}/light.joblib")
    json.dump({"heavy_auc": aucs, "light_click_auc": round(float(lauc), 3),
               "label_rates": {h: round(float(Y[:, j].mean()), 4) for j, h in enumerate(W.HEADS)},
               "trained_on": int(split), "held_out": int(len(X) - split),
               "positives_held_out": {h: int(Y[split:, j].sum()) for j, h in enumerate(W.HEADS)}},
              open(f"{W.DATA}/models.json", "w"))


def load_redis():
    w = np.load(f"{W.DATA}/world.npz")
    t0 = time.time()
    R.flushdb()
    items = np.column_stack([w[f] for f in W.ITEM_FIELDS]).astype(np.float32)
    users = np.hstack([w["taste"], w["personal"]]).astype(np.float32)
    n_items, n_users = len(items), len(users)
    for start in range(0, n_items, 50_000):
        R.mset({f"i:{i}": items[i].tobytes() for i in range(start, min(start + 50_000, n_items))})
    for start in range(0, n_users, 50_000):
        R.mset({f"u:{u}": users[u].tobytes() for u in range(start, min(start + 50_000, n_users))})
    R.set("trending", np.load(f"{W.DATA}/trending.npy").tobytes())
    R.set("seeded", CONFIG)
    log(f"redis loaded: {n_items:,} item records, {n_users:,} user records ({time.time() - t0:.1f}s)")


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
