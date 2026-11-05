#!/usr/bin/env python3
"""Validate and summarize the paired online-protection experiment."""
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
 if len(sys.argv)!=4: raise SystemExit('usage: summarize_online_protection.py <protocol.json> <config.cfg> <evidence-directory>')
 plan=runner.read_json(sys.argv[1]);cfg=runner.config_parse(Path(sys.argv[2]).read_text());root=Path(sys.argv[3]);seeds={str(x) for x in cfg['seeds']}
 if plan.get('schema_version')!=1 or len(seeds)!=30:raise ValueError('unsupported protocol')
 metrics={}
 with (root/'metrics.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','phase','measurement','model','correct','trials','accuracy','logloss']:raise ValueError('unexpected metrics header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','phase','measurement','model'));s,fam,length,phase,measurement,model=key
   if s not in seeds or fam not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('1','2','3') or measurement not in ('online','first_100','probe_a','probe_b') or model not in plan['arms'] or key in metrics:raise ValueError(f'unexpected observation {key}')
   n=int(r['trials']);correct=int(r['correct']);p=float(r['accuracy']);loss=float(r['logloss']);expected=int(length) if measurement=='online' else (100 if measurement=='first_100' else cfg['test_tokens'])
   if n!=expected or not 0<=correct<=n or not math.isfinite(p) or abs(p-correct/n)>5.1e-9 or not math.isfinite(loss) or loss<0:raise ValueError(f'invalid observation {key}')
   metrics[key]=correct/n
 expected=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*3*4*len(plan['arms'])
 if len(metrics)!=expected:raise ValueError('incomplete metrics')
 suppression={}
 with (root/'suppression.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','model','eligible_steps','suppressed_updates','training_steps']:raise ValueError('unexpected suppression header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','model'));s,fam,length,model=key
   if s not in seeds or fam not in plan['families'] or int(length) not in plan['phase_lengths'] or model not in ('protect_inactive','random_slot_control') or key in suppression:raise ValueError(f'unexpected suppression {key}')
   eligible,skips,steps=(int(r[x]) for x in ('eligible_steps','suppressed_updates','training_steps'))
   if steps!=3*int(length) or not 0<=eligible==skips<=steps:raise ValueError(f'invalid suppression {key}')
   suppression[key]=skips/steps
 if len(suppression)!=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*2:raise ValueError('incomplete suppression')
 result={'schema_version':1,'seed_count':len(seeds),'bootstrap':'10,000 seed-paired percentile resamples, seed 20261105, descriptive and unadjusted','conditions':{}}
 for fam in sorted(plan['families']):
  for length in plan['phase_lengths']:
   length=str(length)
   freeze={m:statistics.mean(suppression[(s,fam,length,m)] for s in seeds) for m in ('protect_inactive','random_slot_control')}
   for label,phase,metric in [('b_recall','3','probe_b'),('a_recall','3','probe_a'),('switch_to_b','2','first_100'),('switch_to_a','3','first_100')]:
    values={m:[metrics[(s,fam,length,phase,metric,m)] for s in sorted(seeds,key=int)] for m in plan['arms']}
    pairs={}
    for ref in ('idea2','random_slot_control'):
     delta=[a-b for a,b in zip(values['protect_inactive'],values[ref])]; rng=random.Random(f'20261105:{fam}:{length}:{label}:{ref}');boot=sorted(statistics.mean(rng.choice(delta) for _ in delta) for _ in range(10000))
     pairs[ref]={'mean_difference':statistics.mean(delta),'median_difference':statistics.median(delta),'wins_losses_ties':[sum(x>0 for x in delta),sum(x<0 for x in delta),sum(x==0 for x in delta)],'bootstrap_95_percentile':[boot[250],boot[9750]]}
    result['conditions'][f'{fam}:{length}:{label}']={'mean_accuracy':{m:statistics.mean(v) for m,v in values.items()},'protection_vs':pairs,'mean_suppressed_step_fraction':freeze}
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
