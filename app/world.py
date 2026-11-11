"""The synthetic world the feed runs on: topics, items, users and how users behave.

Everything here is invented and seeded. The ground-truth behaviour functions are
what a real system never sees. The services only see logged labels and the
features computed from them, exactly as a production feed would.
"""
import os

import numpy as np

N_ITEMS = int(os.environ.get("N_ITEMS", 1_000_000))
N_USERS = int(os.environ.get("N_USERS", 100_000))
N_TOPICS = 24
DIM = 64
SEED = 7
DATA = os.environ.get("DATA_DIR", "/data")

# Ranker features, in the order every service uses them.
FEATURES = ["dot", "taste", "sess_engaged", "sess_hidden", "item_ctr", "bait_score", "log_age_h", "log_pop"]
HEADS = ["click", "finish", "like", "hide"]
LIGHT_FEATURES = ["dot", "item_ctr", "log_pop"]          # what the light ranker may use, cheap and per item

# Item record stored in Redis, float32 each.
ITEM_FIELDS = ["topic", "item_ctr", "bait_score", "log_age_h", "log_pop"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def topic_centres(rng):
    c = rng.normal(size=(N_TOPICS, DIM)).astype(np.float32)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def mood(taste_aff, sess_engaged, sess_hidden):
    """How much the user wants this topic right now. taste_aff is the share of
    their long-term taste on the item's topic, scaled so 1.0 is average."""
    return taste_aff + 0.3 * sess_engaged - 1.5 * sess_hidden


def true_probs(taste_aff, sess_engaged, sess_hidden, quality, bait):
    """Ground truth. Unconditional probabilities for each head."""
    m = mood(taste_aff, sess_engaged, sess_hidden)
    m = np.clip(m, -3, 4)
    p_click = sigmoid(-5.6 + 0.55 * m + 2.0 * quality + 1.3 * bait)
    p_finish = p_click * sigmoid(-0.8 + 0.35 * m + 2.5 * quality - 2.8 * bait)
    p_like = p_click * sigmoid(-3.2 + 0.35 * m + 2.5 * quality - 1.0 * bait)
    p_hide = sigmoid(-4.2 + 2.2 * bait - 0.45 * m + 0.9 * sess_hidden)
    return p_click, p_finish, p_like, p_hide


def make_world():
    """Items and users. Returns plain numpy arrays, all seeded."""
    rng = np.random.default_rng(SEED)
    centres = topic_centres(rng)
    topic = rng.integers(N_TOPICS, size=N_ITEMS).astype(np.int16)
    vecs = centres[topic] + 0.12 * rng.normal(size=(N_ITEMS, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    quality = rng.beta(2, 5, N_ITEMS).astype(np.float32)
    bait = (rng.random(N_ITEMS) < 0.12)
    age_h = rng.uniform(0, 90 * 24, N_ITEMS).astype(np.float32)          # uploaded in the last 90 days
    # Popularity: impressions so far, older and better items have more.
    pop = rng.lognormal(np.log(50) + 2.5 * quality + 0.6 * bait, 1.0) * np.minimum(age_h, 240) / 240
    # The content model's bait score: a noisy guess at bait from the thumbnail and title.
    bait_score = np.clip(0.75 * bait + rng.normal(0.15, 0.15, N_ITEMS), 0, 1).astype(np.float32)
    # Historical click rate on an average user, noisy where the item has few impressions.
    base_ctr = sigmoid(-5.6 + 0.55 + 2.0 * quality + 1.3 * bait)
    item_ctr = (rng.binomial(np.maximum(pop, 1).astype(np.int64), base_ctr) + 1) / (np.maximum(pop, 1) + 20)

    taste = rng.dirichlet(np.full(N_TOPICS, 0.12), size=N_USERS).astype(np.float32)
    personal = rng.normal(size=(N_USERS, DIM)).astype(np.float32)     # what makes two users with one taste differ
    personal /= np.linalg.norm(personal, axis=1, keepdims=True)
    return {
        "centres": centres, "topic": topic, "vecs": vecs, "quality": quality, "bait": bait,
        "age_h": age_h, "log_pop": np.log1p(pop).astype(np.float32),
        "log_age_h": np.log1p(age_h).astype(np.float32), "bait_score": bait_score,
        "item_ctr": item_ctr.astype(np.float32), "taste": taste, "personal": personal,
    }


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
