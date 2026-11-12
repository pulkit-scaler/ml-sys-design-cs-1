"""The world the feed runs on: real videos and users from KuaiRand-1K (see kuairand.py).

The services only see what a production feed would see. Item features computed from logs,
each user's taste from their history, session features from the event stream, and models
trained on logged labels.
"""
import os

import numpy as np

N_ITEMS = int(os.environ.get("N_ITEMS", 1_000_000))     # videos sampled from KuaiRand-1K's 4.37 million
N_TOPICS = 24                                           # the 23 commonest first tags (one is "untagged"), and "other"
DIM = 64
SEED = 7
DATA = os.environ.get("DATA_DIR", "/data")

# Ranker features, in the order every service uses them.
FEATURES = ["dot", "taste", "sess_engaged", "sess_hidden", "item_ctr", "skip_rate", "log_age_h", "log_pop"]
HEADS = ["click", "finish", "like", "hide"]             # KuaiRand: valid play, played to the end, like, hate
LIGHT_FEATURES = ["dot", "item_ctr", "log_pop"]          # what the light ranker may use, cheap and per item

# Item record stored in Redis, float32 each.
ITEM_FIELDS = ["topic", "item_ctr", "skip_rate", "log_age_h", "log_pop"]


def topic_centres(rng):
    c = rng.normal(size=(N_TOPICS, DIM)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def make_world():
    """The catalogue, the users and the real training log. Seeded, so every build is the same."""
    import kuairand
    rng = np.random.default_rng(SEED)
    return kuairand.build(N_ITEMS, N_TOPICS, DIM, SEED, topic_centres(rng))


def user_tower_batch(taste, sess_engaged, sess_hidden, personal, centres):
    """Stand-in for the user tower: long-term taste plus this session, mapped to the
    item space, plus a personal offset. Topic arguments are (n, N_TOPICS), personal
    is (n, DIM). Returns (n, DIM) unit vectors."""
    w = np.clip(taste + 0.04 * sess_engaged - 0.25 * sess_hidden, 0, None).astype(np.float32)
    w[w.sum(axis=1) <= 0] = 1.0 / N_TOPICS
    v = np.dot(w, centres)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) + 0.6 * personal
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def user_tower(taste, sess_engaged, sess_hidden, personal, centres):
    return user_tower_batch(taste[None], sess_engaged[None], sess_hidden[None], personal[None], centres)[0]
