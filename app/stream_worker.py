"""Stream job: user actions from Kafka become session features in Redis.

  click, finish, like  ->  s:{user}  e:{topic} += 1   (engaged with this topic this session)
  hide                 ->  s:{user}  h:{topic} += 1   (hid this topic this session)
  finish               ->  s:{user}  last = item      (seed for the more-like-last source)

Session features expire 30 minutes after the user's last action.
"""
import json
import os
import signal
import time

import redis
from confluent_kafka import Consumer
from prometheus_client import Counter, Histogram, start_http_server

R = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379)
LAG = Histogram("stream_event_to_feature_seconds", "From the app sending an action to the feature being written",
                buckets=[.005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 30, 60, 120, 300])
DONE = Counter("stream_events_processed_total", "Actions turned into session features")


def main():
    start_http_server(8000)
    c = Consumer({"bootstrap.servers": os.environ.get("KAFKA", "redpanda:9092"), "group.id": "session-features",
                  "auto.offset.reset": "earliest", "enable.auto.commit": True, "session.timeout.ms": 6000})
    c.subscribe(["events"])
    running = {"on": True}

    def stop(*_):                                  # leave the consumer group cleanly on docker stop,
        running["on"] = False                      # or the next worker waits out the old one's session

    signal.signal(signal.SIGTERM, stop)
    print("stream worker consuming 'events'", flush=True)
    while running["on"]:
        msg = c.poll(0.5)
        if msg is None or msg.error():
            continue
        e = json.loads(msg.value())
        key = f"s:{e['user']}"
        p = R.pipeline(transaction=False)
        if e["action"] == "hide":
            p.hincrbyfloat(key, f"h:{e['topic']}", 1)
        else:
            p.hincrbyfloat(key, f"e:{e['topic']}", 1)
        if e["action"] == "finish":
            p.hset(key, "last", e["item"])
        p.expire(key, 1800)
        lag = time.time() - e["ts"]
        p.incr("stream:processed")
        p.set("stream:last_lag_ms", round(lag * 1e3, 1))
        p.execute()
        LAG.observe(lag)
        DONE.inc()
    c.close()
    print("stream worker stopped, left the consumer group", flush=True)


if __name__ == "__main__":
    main()
