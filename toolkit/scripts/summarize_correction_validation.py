#!/usr/bin/env python3
"""Validate matched correction-ablation evidence and summarize paired outcomes."""
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from toolkit.coordinator import runner

def main():
 if len(sys.argv)!=4: raise SystemExit('usage: summarize_correction_validation.py <protocol.json> <config.cfg> <metrics.csv>')
 plan=runner.read_json(sys.argv[1]);cfg=runner.config_parse(Path(sys.argv[2]).read_text());seeds={str(s) for s in cfg['seeds']}; arms={plan['candidate'],*plan['controls']}
 if plan.get('schema_version')!=1 or len(seeds)!=30 or plan['bootstrap']['draws']!=10000:raise ValueError('unsupported protocol')
 rows={}
 with Path(sys.argv[3]).open(newline='',encoding='utf-8') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','phase','measurement','model','correct','trials','accuracy','logloss']:raise ValueError('unexpected metrics header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','phase','measurement','model'));seed,family,length,phase,measure,model=key
   if seed not in seeds or family not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('1','2','3') or measure not in ('online','first_100','probe_a','probe_b') or model not in arms or key in rows:raise ValueError(f'unexpected observation: {key}')
   n=int(r['trials']);correct=int(r['correct']);acc=float(r['accuracy']);loss=float(r['logloss']);expected=int(length) if measure=='online' else (100 if measure=='first_100' else cfg['test_tokens'])
   if n!=expected or not 0<=correct<=n or not math.isfinite(acc) or abs(acc-correct/n)>5.1e-9 or not math.isfinite(loss) or loss<0:raise ValueError(f'invalid observation: {key}')
   rows[key]=(correct/n,loss)
 expected=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*3*4*len(arms)
 if len(rows)!=expected:raise ValueError(f'incomplete evidence: {len(rows)} != {expected}')
 output={'schema_version':1,'seed_count':len(seeds),'bootstrap':'10,000 paired seed percentile resamples, descriptive and unadjusted','conditions':{}}
 for family in sorted(plan['families']):
  for length in plan['phase_lengths']:
   length=str(length)
   for label,phase,measure in [('switch_to_b','2','first_100'),('switch_to_a','3','first_100'),('final_b','3','probe_b'),('final_a','3','probe_a'),('phase_b_online','2','online')]:
    vals={m:[rows[(s,family,length,phase,measure,m)] for s in sorted(seeds,key=int)] for m in arms}
    pairs={}
    for control in sorted(plan['controls']):
     diffs=[a[0]-b[0] for a,b in zip(vals[plan['candidate']],vals[control])];rng=random.Random(f"{plan['bootstrap']['seed']}:{family}:{length}:{label}:{control}");boot=sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(plan['bootstrap']['draws']))
     pairs[control]={'mean_difference':statistics.mean(diffs),'median_difference':statistics.median(diffs),'wins_losses_ties':[sum(x>0 for x in diffs),sum(x<0 for x in diffs),sum(x==0 for x in diffs)],'bootstrap_95_percentile':[boot[250],boot[9750]]}
    output['conditions'][f'{family}:{length}:{label}']={'mean_accuracy':{m:statistics.mean(x[0] for x in v) for m,v in vals.items()},'mean_logloss':{m:statistics.mean(x[1] for x in v) for m,v in vals.items()},'guarded_vs':pairs}
 print(json.dumps(output,indent=2,sort_keys=True))
if __name__=='__main__':main()
