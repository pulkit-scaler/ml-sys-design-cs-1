# Feed lab: a small, real home-feed system

The live demo for the ML System Design case study **Personalised Recommendation and
Feed Ranking**. The class notebook
explains the design. This lab runs it, so learners can watch the design's
trade-offs happen on a working system. The notebook's last numbered section
before the synthesis is the script for the demo, and `recorded/` holds the run
it quotes.

Everything is synthetic and seeded, and every version is pinned. A rebuild gives
the same world, index, models and pages, and a different kind of chip (Intel
against ARM) gives nearly the same. Only the milliseconds differ between machines.

## Pre-read

[The Feed Lab Toolkit](https://pulkit-scaler.github.io/ml-sys-design-cs-1/) explains the four
tools in this lab that are new to most learners, Redis, Kafka, Prometheus and Grafana, with
real output from the running system. About a 20-minute read. Its source is `docs/index.html`.

## What runs

| Service | What it is in the design | Built from |
|---|---|---|
| `feed` | The feed service. Every hop of the request path, timed | FastAPI |
| `retrieval` | Candidate generation. Two-tower search, new items, trending | Faiss IVF index, 1M items |
| `ranker` | The full multi-task ranking model, four heads | scikit-learn MLP |
| `stream-worker` | Turns user actions into session features | Kafka consumer |
| `redis` | Online feature store for users, items and sessions | Redis 7.4 |
| `redpanda` | The event stream and the impression log | Kafka API |
| `prometheus`, `grafana` | Metrics and the dashboard | Prometheus, Grafana |
| `bootstrap` | One-off job. Builds the world, the index and the models | runs once |

## What differs from production

| In this lab | In production |
|---|---|
| 1 million items, 64-number vectors | Hundreds of millions, sharded indexes on many machines |
| User tower is a fixed formula of taste and session | A trained network over many features |
| Ranker is a small MLP trained on 240,000 simulated impressions | A large network retrained daily or hourly on billions |
| Trending list and item features computed once at start | Recomputed by batch jobs every few minutes or hours |
| One replica of each service, no autoscaling | Many replicas behind load balancers, across zones |
| Ground-truth user behaviour comes from a formula | Real people |

The design is the same one the notebook draws. The scale is not. Read every
millisecond as "this machine, this size", and use the vectors-scored column to
reason about larger catalogues.

## Requirements

- Docker with Compose v2 (`docker compose version` prints v2 or later).
  Docker Desktop on macOS or Windows, or Docker Engine on Linux. Intel and ARM both work.
- At least 6 CPUs and 3 GB of memory given to Docker. Lower `N_ITEMS` and the CPU
  settings in `.env` for smaller machines.
- Python 3.9 or later on the machine that runs `lab.py`, standard library only.
  On Windows, use `py` where this page says `python3`.
- Network access the first time, to pull images and packages. After that it runs offline.

## Start, check, stop

```bash
git clone https://github.com/pulkit-scaler/ml-sys-design-cs-1.git
cd ml-sys-design-cs-1
cp .env.example .env          # optional, every setting has a default (Windows: copy .env.example .env)
docker compose up -d --build  # first run downloads images and builds the world, a few minutes
python3 lab.py status         # all services up, and the dashboard address
python3 lab.py smoke          # runs every demo and checks it, about a minute
```

Stop with `docker compose stop`, which keeps the built world for next time.
`docker compose down -v` removes everything, including the built world.

Run `smoke` before every class. If it prints `SMOKE PASSED`, every demo below works.

## The demos

The notebook has the full script, with what to say and the chat questions.

| Demo | Command | What learners see |
|---|---|---|
| One request, hop by hop | `python3 lab.py trace` | Each hop's time against the 200 ms budget, and the page |
| Recall against work | `python3 lab.py sweep`, then `python3 lab.py nprobe N` under load | The recall/latency curve on a real index |
| A feed that reacts | `python3 lab.py fresh`, then again with `docker compose stop stream-worker` | Hides change the next page, until the stream stops |
| When the ranker fails | `python3 lab.py load` in one terminal, `docker compose stop ranker` and `pause ranker` in another | Fallback with no errors, and the cost of a hung dependency |

`python3 lab.py drill` runs the failure demo by itself, for a backup or a recording.

## Settings

All in `.env`. See `.env.example` for the full list with comments.

| Setting | Default | What it changes |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1` | `0.0.0.0` to open the dashboard from another machine |
| `FEED_PORT`, `GRAFANA_PORT`, ... | 8080, 3000, ... | Change any that are taken on your machine |
| `N_ITEMS`, `N_USERS` | 1,000,000 and 100,000 | Size of the world |
| `*_CPUS` | 1, 2, 1 | CPU limits, so timings compare across machines |
| `NPROBE` | 128 | Clusters searched per query at start |
| `RANKER_TIMEOUT_MS` | 80 | The ranker's deadline before the fallback |

After changing `.env`, run `docker compose up -d` again. Changing `N_ITEMS` or
`N_USERS` rebuilds the world on the next start.

## Running it on a cloud VM

Any Linux VM with Docker, 8 vCPUs and 8 GB is plenty. In `.env`, set
`BIND_ADDR=0.0.0.0` and open the Grafana port in the firewall, or keep
`127.0.0.1` and use an SSH tunnel, for example `ssh -L 3000:localhost:3000 -L 8080:localhost:8080 vm`.
Run `lab.py` on the VM itself. To drive the read-only demos from your own
machine instead, set `LAB_HOST` to the VM's address, for example
`LAB_HOST=10.0.0.5 python3 lab.py trace`.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `port is already allocated` | Change that port in `.env` |
| `bootstrap` exits with code 137 | Docker ran out of memory. Give it more, or lower `N_ITEMS` |
| `lab.py status` says DOWN | `docker compose ps -a` and `docker compose logs <service>` |
| Smoke fails at the stream step | `docker compose restart stream-worker` and run it again |
| Dashboard shows "No data" | Send traffic with `python3 lab.py load`. Panels fill within 15 seconds |
| A demo was left half done | `docker compose unpause ranker; docker compose start ranker stream-worker` |
