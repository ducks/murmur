#!/usr/bin/env python3
"""Validate post-hoc correction guard traces and summarize across seeds."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from toolkit.coordinator import runner

def main():
 if len(sys.argv)!=4:raise SystemExit('usage: summarize_guard_diagnostics.py <protocol.json> <config.cfg> <guard-diagnostics.csv>')
 plan=runner.read_json(sys.argv[1]);cfg=runner.config_parse(Path(sys.argv[2]).read_text());seeds={str(x) for x in cfg['seeds']};groups=defaultdict(list);seen=set()
 with Path(sys.argv[3]).open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','phase','window','mean_guard','mean_abs_correction','corrections']:raise ValueError('unexpected diagnostic header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','phase','window'));seed,family,length,phase,window=key
   if seed not in seeds or family not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('1','2','3') or window not in ('first_100','remaining') or key in seen:raise ValueError(f'unexpected diagnostic {key}')
   guard=float(r['mean_guard']);correction=float(r['mean_abs_correction']);n=int(r['corrections']);expected=(99 if phase=='1' else 100) if window=='first_100' else int(length)-100
   if n!=expected or not math.isfinite(guard) or not 0<=guard<=1 or not math.isfinite(correction) or correction<0:raise ValueError(f'invalid diagnostic {key}')
   seen.add(key);groups[(family,length,phase,window)].append((guard,correction))
 expected=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*3*2
 if len(seen)!=expected:raise ValueError(f'incomplete diagnostics: {len(seen)} != {expected}')
 out={'schema_version':1,'qualification':'Post-hoc summaries of per-seed window means; not per-event extrema or a causal intervention.','conditions':{}}
 for (family,length,phase,window),values in sorted(groups.items()):
  guards=[x[0] for x in values];out['conditions'][f'{family}:{length}:{phase}:{window}']={'seeds':len(values),'mean_guard':statistics.mean(guards),'min_seed_window_mean_guard':min(guards),'max_seed_window_mean_guard':max(guards),'mean_abs_correction':statistics.mean(x[1] for x in values)}
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
