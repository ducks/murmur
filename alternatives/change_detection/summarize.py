#!/usr/bin/env python3
"""Summarize per-seed CSV files produced by main.rs; stdlib only."""
import csv
import sys

rows = []
for filename in sys.argv[1:]:
    with open(filename, newline="") as f:
        rows.extend(csv.DictReader(f))
if not rows:
    raise SystemExit("usage: summarize.py file.csv [file.csv ...]")
for detector in ("bocpd", "cusum"):
    thresholds = sorted({float(r["threshold"]) for r in rows if r["detector"] == detector})
    for threshold in thresholds:
        entries = []
        for case in ("stable10", "stable25", "switch50", "switch100", "switch200"):
            subset = [r for r in rows if r["detector"] == detector and float(r["threshold"]) == threshold and r["case"] == case]
            if case.startswith("stable"):
                entries.append(f"{case}: {sum(r['any_alarm'] == 'true' for r in subset)}/{len(subset)} alarms")
            else:
                entries.append(f"{case}: {sum(int(r['switches_detected']) for r in subset)}/{sum(int(r['switches_total']) for r in subset)} switches")
        print(f"{detector} {threshold:g}: " + "; ".join(entries))
