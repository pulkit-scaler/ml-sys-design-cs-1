# Feed lab: a small, real home-feed system

The live demo for the ML System Design case study **Personalised Recommendation and
Feed Ranking**. The class notebook
explains the design. This lab runs it, so learners can watch the design's
trade-offs happen on a working system. The notebook's last numbered section
before the synthesis is the script for the demo, and `recorded/` holds the run
it quotes.

## The data

The feed serves real data, [KuaiRand-1K](https://kuairand.com/), logs from the Kuaishou
short-video app published by Gao et al. at CIKM 2022
([paper](https://arxiv.org/abs/2208.08696)). It holds 1,000 real users and every video they
were shown in April and May 2022, 4.37 million videos. The class notebook reads the same
users from the dataset's Pure version. Every video those users were shown at random, 7,388 of the
7,583 the notebook reads, is in this lab's catalogue. The other 195 were never shown to them at random.

- **Real.** The 1,000 users and their tastes, taken from what each one watched. A seeded
  sample of 1 million videos with their real tags, upload dates, play rates, skip rates and
  popularity. The rankers train on 240,000 real impressions from the app's own log, with
  real labels. Click is a valid play, finish is watching to the end, like is a like and
  hide is KuaiRand's "hate".
- **Built.** The user and item vectors. A trained two-tower model would learn them. Here
  they come from each video's tag, author and play rate, and each user's history.

The first start downloads the dataset from Zenodo (1.1 GB, checksum checked) into
`./kuairand/` on your machine, so `docker compose down -v` does not delete it. To skip the
download, put `KuaiRand-1K.tar.gz` in that folder yourself. The dataset is licensed
CC BY-SA 4.0 by its authors.

Every version is pinned and every random choice seeded. A rebuild gives the same world,
index, models and pages, and a different kind of chip (Intel against ARM) gives nearly
the same. Only the milliseconds differ between machines.

## Pre-read

[The Feed Lab Toolkit](https://pulkit-scaler.github.io/ml-sys-design-cs-1/) explains the four
tools in this lab that are new to most learners, Redis, Kafka, Prometheus and Grafana, with
real output from the running system. About a 20-minute read. Its source is `docs/index.html`.

## What runs

| Service | What it is in the design | Built from |
|---|---|---|
| `feed` | The feed service. Every hop of the request path, timed | FastAPI |
| `retrieval` | Candidate generation. Two-tower search, new items, trending | Faiss IVF index, 1M real videos |
| `ranker` | The full multi-task ranking model, four heads | scikit-learn MLP |
| `stream-worker` | Turns user actions into session features | Kafka consumer |
| `redis` | Online feature store for users, items and sessions | Redis 7.4 |
| `redpanda` | The event stream and the impression log | Kafka API |
| `prometheus`, `grafana` | Metrics and the dashboard | Prometheus, Grafana |
| `bootstrap` | One-off job. Reads KuaiRand, builds the index, trains the models | runs once |

## What differs from production

| In this lab | In production |
|---|---|
| 1 million videos, 1,000 users, 64-number vectors | Hundreds of millions, sharded indexes on many machines |
| User and item towers are fixed formulas over real taste, tags, authors and play rates | Trained networks over many features |
| Ranker is a small MLP trained on 240,000 real impressions | A large network retrained daily or hourly on billions |
| Item counts are KuaiRand's averages over the whole month, so a training impression sees a little of its own future, and ages are taken at the end of the log | Counts and ages as they stood at each impression, recomputed every few minutes or hours |
| One replica of each service, no autoscaling | Many replicas behind load balancers, across zones |
| Users in demos are driven by `lab.py`, so the hides are scripted | Real people, acting on what they see |

The design is the same one the notebook draws. The scale is not. Read every
millisecond as "this machine, this size", and use the vectors-scored column to
reason about larger catalogues.

## Requirements

- Docker with Compose v2 (`docker compose version` prints v2 or later).
  Docker Desktop on macOS or Windows, or Docker Engine on Linux. Intel and ARM both work.
- At least 6 CPUs and 4 GB of memory given to Docker. The one-off build peaks near 2.8 GB,
  because it reads the whole dataset whatever `N_ITEMS` is. Lower the CPU settings in `.env`
  for smaller machines.
- Python 3.9 or later on the machine that runs `lab.py`, standard library only.
  On Windows, use `py` where this page says `python3`.
- Network access the first time, to pull images and packages and download the 1.1 GB dataset.
  After that it runs offline. About 6 GB of free disk, 2 GB of it for the dataset and the rest
  for the images.

## Start, check, stop

```bash
git clone https://github.com/pulkit-scaler/ml-sys-design-cs-1.git
cd ml-sys-design-cs-1
cp .env.example .env          # optional, every setting has a default (Windows: copy .env.example .env)
docker compose up -d --build  # first run downloads images and the dataset, then builds the world
python3 lab.py status         # all services up, and the dashboard address
python3 lab.py smoke          # runs every demo and checks it, about a minute
```

Stop with `docker compose stop`, which keeps the built world for next time.
`docker compose down -v` removes everything, including the built world.

Run `smoke` before every class. If it prints `SMOKE PASSED`, every demo below works.
In class, `python3 lab.py show` walks you through them in order (see the guided showcase below).

## The demos

The notebook has the full script, with what to say and the chat questions.

| Demo | Command | What learners see |
|---|---|---|
| One request, hop by hop | `python3 lab.py trace` | Each hop's time against the 200 ms budget, and the page |
| Recall against work | `python3 lab.py sweep`, then `python3 lab.py nprobe N` under load | The recall/latency curve on a real index |
| A feed that reacts | `python3 lab.py fresh`, then again with `docker compose stop stream-worker` | Hides change the next page, until the stream stops |
| When the ranker fails | `python3 lab.py load` in one terminal, `docker compose stop ranker` and `pause ranker` in another | Fallback with no errors, and the cost of a hung dependency |

`python3 lab.py drill` runs the failure demo by itself, for a backup or a recording.

## Guided showcase for the instructor

`python3 lab.py show` runs every demo above in class order, one step at a time. For each
step it prints what to put on screen and the question to post in the chat, then waits.
Press Enter to run the demo, `s` to skip it or `q` to stop. After each demo it prints
PASS or FAIL checks, a line that reads the live numbers, and the teaching point.

```bash
python3 lab.py show --list              # the six steps, with notebook section and minutes
python3 lab.py show                     # the whole class run, Enter between steps
python3 lab.py show --only fresh,stale  # some steps, for example to rehearse one
python3 lab.py show --auto              # no pauses, about two minutes, for a rehearsal or a recording
```

| Step | Notebook | What the checks test |
|---|---|---|
| `trace` | 11.2 | A full page of 20 from the full ranker, every hop's p99 inside its budget, the request's p99 inside 175 ms |
| `sweep` | 11.3 | Recall rises with nprobe, nprobe 256 (the default) finds at least 0.9 of the true top 100, 1,024 clusters cost more vectors than 256 |
| `nprobe` | 11.3, optional | Under load, 8 times the clusters makes each search slower, read from the "ANN search time" histogram |
| `fresh` | 11.4 | Three hides become features in Redis, and the hidden topic shrinks on the next page |
| `stale` | 11.4 | With the stream worker stopped, the hides wait in Kafka and the hidden topic does not shrink, with no error. Restarted, it catches up |
| `drill` | 11.5 | Ranker stopped, the fallback serves with no errors. Ranker hung, still no errors, but p50 jumps past the 80 ms deadline |

The show stops and restarts containers as the demos need, and always leaves the
ranker and the stream worker running and nprobe as it found it, even after Ctrl-C.
A failed check does not end the show. It says which demo failed, so you can talk
through that demo's recorded run in the notebook instead. `recorded/07_show.txt` is
a full `--auto` run. Live numbers differ from it run to run, which is why the checks
test the direction of each effect rather than exact values.

## Settings

All in `.env`. See `.env.example` for the full list with comments.

| Setting | Default | What it changes |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1` | `0.0.0.0` to open the dashboard from another machine |
| `FEED_PORT`, `GRAFANA_PORT`, ... | 8080, 3000, ... | Change any that are taken on your machine |
| `KUAIRAND_DIR` | `./kuairand` | Where the dataset is downloaded and kept |
| `N_ITEMS` | 1,000,000 | How many of KuaiRand's videos the feed serves |
| `*_CPUS` | 1, 2, 1 | CPU limits, so timings compare across machines |
| `NPROBE` | 256 | Clusters searched per query at start, set from the sweep |
| `RANKER_TIMEOUT_MS` | 80 | The ranker's deadline before the fallback |

After changing `.env`, run `docker compose up -d` again. Changing `N_ITEMS` rebuilds
the world on the next start.

## Running it on a cloud VM

Any Linux VM with Docker, 8 vCPUs and 8 GB is plenty. In `.env`, set
`BIND_ADDR=0.0.0.0` and open the Grafana port in the firewall, or keep
`127.0.0.1` and use an SSH tunnel, for example `ssh -L 3000:localhost:3000 -L 8080:localhost:8080 vm`.
Run `lab.py` on the VM itself, and always for `smoke`, `drill` and `show`, which stop and start containers. To drive the read-only demos from your own
machine instead, set `LAB_HOST` to the VM's address, for example
`LAB_HOST=10.0.0.5 python3 lab.py trace`.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `port is already allocated` | Change that port in `.env` |
| `bootstrap` exits with code 137 | Docker ran out of memory. Give Docker at least 4 GB. Lowering `N_ITEMS` helps little, since the whole dataset is read |
| `bootstrap` says checksum mismatch | A download was cut short. Delete `kuairand/KuaiRand-1K.tar.gz.part` and start again |
| `lab.py status` says DOWN | `docker compose ps -a` and `docker compose logs <service>` |
| Smoke fails at the stream step | `docker compose restart stream-worker` and run it again |
| Dashboard shows "No data" | Send traffic with `python3 lab.py load`. Panels fill within 15 seconds |
| A demo was left half done | `docker compose unpause ranker; docker compose start ranker stream-worker` |
