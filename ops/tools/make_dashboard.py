"""Writes ops/grafana/dashboards/feed.json. Run from the lab folder: python3 ops/tools/make_dashboard.py"""
import json

ds = {"type": "prometheus", "uid": "prom"}


def ts(pid, title, targets, x, y, w=12, h=8, unit="ms", desc="", overrides=()):
    return {"id": pid, "type": "timeseries", "title": title, "description": desc, "datasource": ds,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit, "min": 0, "custom": {"lineWidth": 2, "fillOpacity": 0}},
                            "overrides": list(overrides)},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
            "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": ds}
                        for i, (e, l) in enumerate(targets)]}


def stat(pid, title, expr, x, y, w=4, h=4, desc="", decimals=1):
    return {"id": pid, "type": "stat", "title": title, "description": desc, "datasource": ds,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": "none", "decimals": decimals}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "none", "graphMode": "none"},
            "targets": [{"refId": "A", "expr": expr, "datasource": ds}]}


def q(p, m, by=""):
    return f"1000 * histogram_quantile({p}, sum by (le{by}) (rate({m}_bucket[15s])))"


def colour(name, c, width=3):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}},
                           {"id": "custom.lineWidth", "value": width}]}


panels = [
    stat(1, "Feed requests / s", "sum(rate(feed_requests_total[15s]))", 0, 0),
    stat(2, "Feed p99 (ms)", q(0.99, "feed_request_seconds"), 4, 0, desc="Server side. The budget is 200 ms"),
    stat(3, "Not served by the full ranker (%)",
         '100 * sum(rate(feed_requests_total{served_by!="full"}[15s])) / sum(rate(feed_requests_total[15s]))', 8, 0),
    stat(4, "IVF nprobe", "max(retrieval_nprobe)", 12, 0, decimals=0),
    stat(5, "Stream backlog (events)", "max(stream_backlog_events)", 16, 0, decimals=0,
         desc="Actions sent to Kafka that the stream worker has not turned into features yet"),
    stat(6, "Event to feature p99 (ms)", q(0.99, "stream_event_to_feature_seconds"), 20, 0,
         desc="From the app sending an action to the session feature being written, last 15 s with events"),
    ts(7, "Feed latency, server side", [(q(0.5, "feed_request_seconds"), "p50"), (q(0.99, "feed_request_seconds"), "p99")],
       0, 4, desc="Compare with the 200 ms p99 budget"),
    ts(8, "How requests were served", [("sum by (served_by) (rate(feed_requests_total[15s]))", "{{served_by}}")],
       12, 4, unit="reqps"),
    ts(9, "p99 per hop", [(q(0.99, "feed_hop_seconds", ", hop"), "{{hop}}")], 0, 12, w=24, h=9,
       desc="The hop-by-hop budget from the notebook, measured", overrides=[colour("ranking_model", "red")]),
    ts(10, "ANN search time (two-tower source)",
       [(q(0.5, "retrieval_search_seconds"), "p50"), (q(0.99, "retrieval_search_seconds"), "p99")], 0, 21),
    ts(11, "Full ranking model, scoring time",
       [(q(0.5, "ranker_score_seconds"), "p50"), (q(0.99, "ranker_score_seconds"), "p99")], 12, 21),
]
dash = {"uid": "feedlab", "title": "Feed lab", "refresh": "5s", "time": {"from": "now-5m", "to": "now"},
        "schemaVersion": 39, "timezone": "browser", "panels": panels, "tags": ["feedlab"]}
with open("ops/grafana/dashboards/feed.json", "w") as f:
    json.dump(dash, f, indent=1)
print(f"dashboard written, {len(panels)} panels")
