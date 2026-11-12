"""Build the feed's world from KuaiRand-1K, real logs from the Kuaishou short-video app.

Source: Gao et al., "KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly
Exposed Videos", CIKM 2022, https://kuairand.com . Zenodo record 10439422, KuaiRand-1K.tar.gz,
md5 6b0b9c8222d67fcd4c676218edca3f1f. Licence CC BY-SA 4.0 (the archive's LICENSE file).

The archive is downloaded once into KUAIRAND_DIR (a folder on the host, so it survives
`docker compose down -v`) and read in a single streaming pass, without unpacking it.

What is real and what is not:
  real   every video's tag, upload date and platform-wide counts (shows, valid plays,
         quick skips), the 1,000 users and what they watched, and every training label.
         The counts are KuaiRand's averages over the whole month, so a training impression
         sees a little of its own future, and ages are taken at the end of the log.
  built  item and user vectors. A trained two-tower model would learn them. Here they are
         built from each video's tag, author and real play rate, and each user's watch
         history, so that similar videos sit close together and good ones score higher.
         They stand in for the towers.
"""
import hashlib
import io
import os
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

URL = "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
MD5 = "6b0b9c8222d67fcd4c676218edca3f1f"
SRC_DIR = os.environ.get("KUAIRAND_DIR", "/kuairand")
TAR = os.path.join(SRC_DIR, "KuaiRand-1K.tar.gz")

NOW = pd.Timestamp("2022-05-09")          # the end of the last logged day
SESSION_GAP_MS = 30 * 60 * 1000           # a session ends after 30 idle minutes, like the stream's TTL
LOOKBACK = 50                             # session features count at most the last 50 actions
AUTHOR_BUCKETS = 1 << 18
QUALITY_IN_LENGTH = 0.2                   # how much a video's play rate stretches its vector


def log(msg):
    print(f"[kuairand {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch():
    """The archive, downloaded once and checked."""
    os.makedirs(SRC_DIR, exist_ok=True)
    if os.path.exists(TAR) and md5(TAR) == MD5:
        log(f"using {TAR}")
        return TAR
    log(f"downloading KuaiRand-1K (1.1 GB) from Zenodo into {SRC_DIR}. This happens once per machine.")
    part = TAR + ".part"
    t0, last = time.time(), 0
    with urllib.request.urlopen(URL, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        for block in iter(lambda: r.read(1 << 22), b""):
            f.write(block)
            done += len(block)
            if time.time() - last > 30:
                log(f"  {done / 1e6:,.0f} of {total / 1e6:,.0f} MB")
                last = time.time()
    if md5(part) != MD5:
        raise SystemExit(f"checksum mismatch for {part}; delete it and run again")
    os.replace(part, TAR)
    log(f"downloaded in {time.time() - t0:.0f}s, checksum ok")
    return TAR


LOG_COLS = {"user_id": "int32", "video_id": "int32", "time_ms": "int64", "is_click": "int8", "is_like": "int8",
            "is_hate": "int8", "long_view": "int8", "play_time_ms": "int64", "duration_ms": "int64"}
WANTED = {
    "video_features_basic_1k.csv": dict(usecols=["video_id", "author_id", "upload_dt", "tag"],
                                        dtype={"video_id": "int32", "author_id": "int64", "tag": "str"}),
    "video_features_statistic_1k.csv": dict(usecols=["video_id", "counts", "show_cnt", "valid_play_cnt", "play_cnt",
                                                     "short_time_play_cnt"]),
    "log_standard_4_08_to_4_21_1k.csv": dict(usecols=list(LOG_COLS), dtype=LOG_COLS),
    "log_standard_4_22_to_5_08_1k.csv": dict(usecols=list(LOG_COLS), dtype=LOG_COLS),
    "log_random_4_22_to_5_08_1k.csv": dict(usecols=["video_id"], dtype={"video_id": "int32"}),
}


class _Member(io.RawIOBase):
    """A tar member read in stream mode, wrapped so pandas can read it (it is not seekable)."""
    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        b[:len(data)] = data
        return len(data)


def read_archive(path):
    """One streaming pass over the archive, parsing only the files and columns needed."""
    out, t0 = {}, time.time()
    with tarfile.open(path, mode="r|gz") as tar:
        for m in tar:
            name = os.path.basename(m.name)
            if name in WANTED:
                out[name] = pd.read_csv(io.BufferedReader(_Member(tar.extractfile(m)), 1 << 20), **WANTED[name])
                log(f"  read {name}: {len(out[name]):,} rows ({time.time() - t0:.0f}s)")
    missing = set(WANTED) - set(out)
    if missing:
        raise SystemExit(f"archive is missing {sorted(missing)}")
    return out


def build(n_items, n_topics, dim, seed, centres):
    """Returns the world arrays and the real training log, ready for bootstrap.py."""
    raw = read_archive(fetch())
    basic = raw["video_features_basic_1k.csv"].sort_values("video_id").reset_index(drop=True)
    stats = raw["video_features_statistic_1k.csv"].set_index("video_id").reindex(basic.video_id)
    history = raw["log_standard_4_08_to_4_21_1k.csv"]
    recent = raw["log_standard_4_22_to_5_08_1k.csv"]
    n_all = len(basic)

    # Topics: each video's first KuaiRand tag. The commonest n_topics - 1 get a topic each, and
    # untagged videos count as one of those "tags" (they are the 14th commonest). Every rarer tag
    # shares the last topic, "other".
    first = basic.tag.fillna("none").str.split(",").str[0]
    common = first.value_counts().index[: n_topics - 1].tolist()
    topic_of_tag = {t: i for i, t in enumerate(common)}
    topic_all = first.map(topic_of_tag).fillna(n_topics - 1).to_numpy().astype(np.int16)
    vid_to_row = np.full(int(basic.video_id.max()) + 1, -1, np.int64)
    vid_to_row[basic.video_id.to_numpy()] = np.arange(n_all)

    # The catalogue: every video KuaiRand showed at random (the pool the class notebook reads),
    # plus a seeded sample of the rest, so popularity and upload dates keep their real spread.
    # The lab calls them items 0..n_items-1.
    rng = np.random.default_rng(seed)
    pool = np.unique(vid_to_row[raw["log_random_4_22_to_5_08_1k.csv"].video_id.to_numpy()])
    rest = np.setdiff1d(np.arange(n_all), pool)
    fill = rng.choice(rest, max(0, min(n_items, n_all) - len(pool)), replace=False)
    pick = np.sort(np.r_[pool, fill])
    item_of_row = np.full(n_all, -1, np.int64)
    item_of_row[pick] = np.arange(len(pick))

    show = (stats.counts * stats.show_cnt).to_numpy()
    item_ctr = np.clip(stats.valid_play_cnt / stats.show_cnt, 0, 1).fillna(0).to_numpy()
    skip_rate = np.clip(stats.short_time_play_cnt / stats.play_cnt, 0, 1).fillna(0).to_numpy()
    up = pd.to_datetime(basic.upload_dt, errors="coerce")
    age_h = ((NOW - up).dt.total_seconds() / 3600 - 12).fillna(((NOW - up).dt.total_seconds() / 3600 - 12).median())
    age_h = np.maximum(age_h.to_numpy(), 1.0)     # upload dates are days, so each video counts as uploaded at noon

    # Item vectors, standing in for the item tower: the topic's centre, plus the author's own
    # direction (videos by one author are alike), plus a little per-video noise.
    authors = (basic.author_id.to_numpy() % AUTHOR_BUCKETS).astype(np.int64)
    author_vec = rng.normal(size=(AUTHOR_BUCKETS, dim)).astype(np.float32)
    author_vec /= np.linalg.norm(author_vec, axis=1, keepdims=True)
    vecs = (centres[topic_all[pick]] + 0.6 * author_vec[authors[pick]]
            + 0.06 * rng.normal(size=(len(pick), dim)).astype(np.float32))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # A trained item tower also learns quality, through each vector's length, so a good video scores
    # higher for everyone. Here the length follows the video's real valid-play rate.
    c = np.clip(item_ctr[pick], 0.01, 0.99)
    z = np.log(c / (1 - c))
    vecs *= (1 + QUALITY_IN_LENGTH * np.clip((z - z.mean()) / z.std(), -2.5, 2.5))[:, None]

    # Users: long-term taste from what each user watched for 18 s or more in the first two weeks,
    # shrunk toward the average. The personal vector points at the authors they watched.
    n_users = int(max(history.user_id.max(), recent.user_id.max())) + 1
    h = history[history.long_view == 1]
    h_rows = vid_to_row[h.video_id.to_numpy()]
    h_topic = topic_all[h_rows]
    counts = np.zeros((n_users, n_topics), np.float64)
    np.add.at(counts, (h.user_id.to_numpy(), h_topic), 1)
    overall = counts.sum(axis=0) / max(counts.sum(), 1)
    taste = ((counts + 3 * overall) / (counts.sum(axis=1, keepdims=True) + 3)).astype(np.float32)
    personal = np.zeros((n_users, dim), np.float32)
    np.add.at(personal, h.user_id.to_numpy(), author_vec[authors[h_rows]])
    norm = np.linalg.norm(personal, axis=1, keepdims=True)
    personal = np.where(norm > 0, personal / np.maximum(norm, 1e-9), 0).astype(np.float32)
    log(f"world: {len(pick):,} of {n_all:,} videos ({len(pool):,} from the random-exposure pool), "
        f"{n_users:,} users, {n_topics} topics "
        f"(tags {', '.join(common[:5])} ... and 'other')")

    world = {
        "centres": centres, "topic": topic_all[pick], "vecs": vecs.astype(np.float32),
        "video_id": basic.video_id.to_numpy()[pick].astype(np.int32),
        "age_h": age_h[pick].astype(np.float32), "log_age_h": np.log1p(age_h[pick]).astype(np.float32),
        "log_pop": np.log1p(show[pick]).astype(np.float32), "item_ctr": item_ctr[pick].astype(np.float32),
        "skip_rate": skip_rate[pick].astype(np.float32), "taste": taste, "personal": personal,
        "topic_tags": np.array(common + ["other"]),
    }
    return world, training_rows(recent, vid_to_row, item_of_row, topic_all, n_topics, seed)


def training_rows(recent, vid_to_row, item_of_row, topic_all, n_topics, seed, n=240_000):
    """Real impressions the old recommender logged, 22 Apr - 8 May 2022, with real labels and the
    session features the stream would have held at that moment (earlier actions in the session)."""
    d = recent.sort_values(["user_id", "time_ms"], kind="stable").reset_index(drop=True)
    rows = vid_to_row[d.video_id.to_numpy()]
    topic = topic_all[rows]
    item = item_of_row[rows]
    finish = ((d.play_time_ms >= d.duration_ms) & (d.duration_ms > 0)).to_numpy()
    engaged = d.is_click.to_numpy() + finish + d.is_like.to_numpy()      # one stream event per action
    hidden = d.is_hate.to_numpy()
    user, t = d.user_id.to_numpy(), d.time_ms.to_numpy()
    new_session = np.r_[True, (user[1:] != user[:-1]) | (np.diff(t) > SESSION_GAP_MS)]
    start = np.maximum.accumulate(np.where(new_session, np.arange(len(d)), 0))

    rng = np.random.default_rng(seed + 1)
    cand = np.flatnonzero(item >= 0)                                     # only videos in the catalogue
    take = np.sort(rng.choice(cand, min(n, len(cand)), replace=False))
    eng = np.zeros((len(take), n_topics), np.float32)
    hid = np.zeros((len(take), n_topics), np.float32)
    for k, p in enumerate(take):
        s = max(start[p], p - LOOKBACK)
        if s < p:
            tp = topic[s:p]
            eng[k] = np.bincount(tp, weights=engaged[s:p], minlength=n_topics)
            hid[k] = np.bincount(tp, weights=hidden[s:p], minlength=n_topics)
    labels = np.column_stack([d.is_click.to_numpy()[take], finish[take], d.is_like.to_numpy()[take],
                              hidden[take]]).astype(np.int8)
    log(f"training log: {len(take):,} real impressions of {len(cand):,} on catalogue videos")
    return {"user": user[take].astype(np.int32), "item": item[take].astype(np.int64), "eng": eng, "hid": hid,
            "labels": labels}
