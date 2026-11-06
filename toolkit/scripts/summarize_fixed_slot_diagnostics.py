#!/usr/bin/env python3
"""Validate and summarize paired fixed-mapping diagnostic evidence."""
import csv,json,math,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from toolkit.coordinator import runner

def summarize(values,seed):
 d=[a-b for a,b in values];r=random.Random(seed);boot=sorted(statistics.mean(r.choice(d) for _ in d) for _ in range(10000))
 return {'correction_mean':statistics.mean(x[0] for x in values),'no_correction_mean':statistics.mean(x[1] for x in values),
         'mean_difference':statistics.mean(d),'median_difference':statistics.median(d),'wins_losses_ties':[sum(x>0 for x in d),sum(x<0 for x in d),sum(x==0 for x in d)],'bootstrap_95_percentile':[boot[250],boot[9750]]}

def main():
 if len(sys.argv)!=4:raise SystemExit('usage: summarize_fixed_slot_diagnostics.py <protocol.json> <config.cfg> <evidence-directory>')
 plan=runner.read_json(sys.argv[1]);cfg=runner.config_parse(Path(sys.argv[2]).read_text());root=Path(sys.argv[3]);seeds={str(x) for x in cfg['seeds']};arms=set(plan['arms'])
 if plan.get('schema_version')!=1 or len(seeds)!=30 or plan['bootstrap']['draws']!=10000:raise ValueError('unsupported protocol')
 metrics={}
 with (root/'metrics.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','phase','measurement','model','correct','trials','accuracy','logloss']:raise ValueError('unexpected metrics header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','phase','measurement','model'));s,family,length,phase,measure,model=key
   if s not in seeds or family not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('1','2','3') or measure not in ('online','first_100','probe_b') or (measure=='probe_b' and phase!='3') or model not in arms or key in metrics:raise ValueError(f'unexpected metric {key}')
   n=int(r['trials']);correct=int(r['correct']);p=float(r['accuracy']);loss=float(r['logloss']);want=int(length) if measure=='online' else (100 if measure=='first_100' else cfg['test_tokens'])
   if n!=want or not 0<=correct<=n or not math.isfinite(p) or abs(p-correct/n)>5.1e-9 or not math.isfinite(loss) or loss<0:raise ValueError(f'invalid metric {key}')
   metrics[key]=correct/n
 if len(metrics)!=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*7*len(arms):raise ValueError('incomplete metrics')
 diag={};mappings={}
 with (root/'diagnostics.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=['seed','family','phase_length','phase','window','model','slot_a','alignment_margin','pre_assignment_error','post_assignment_error','b_memory_mse','trials']:raise ValueError('unexpected diagnostics header')
  for r in reader:
   key=tuple(r[x] for x in ('seed','family','phase_length','phase','window','model'));s,family,length,phase,window,model=key
   if s not in seeds or family not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('2','3') or window not in ('first_100','remaining') or model not in arms or key in diag:raise ValueError(f'unexpected diagnostic {key}')
   slot=int(r['slot_a']);margin=float(r['alignment_margin']);values=[float(r[x]) for x in ('pre_assignment_error','post_assignment_error','b_memory_mse')];n=int(r['trials'])
   if slot not in (0,1) or not math.isfinite(margin) or margin<0 or any(not math.isfinite(x) or not 0<=x<=1 for x in values) or n!=(100 if window=='first_100' else int(length)-100):raise ValueError(f'invalid diagnostic {key}')
   mapping_key=(s,family,length,model); prior=mappings.setdefault(mapping_key,(slot,margin))
   if prior!=(slot,margin):raise ValueError(f'mapping changed: {mapping_key}')
   diag[key]=values
 if len(diag)!=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*2*2*len(arms):raise ValueError('incomplete diagnostics')
 out={'schema_version':1,'seeds':len(seeds),'qualification':'Oracle-assisted mapping fixed after A; descriptive seed-paired intervals, unadjusted for multiple comparisons. Lower diagnostic error is better.','conditions':{}}
 for family in sorted(plan['families']):
  for length in plan['phase_lengths']:
   length=str(length)
   for label,phase,measure in [('switch_to_b','2','first_100'),('switch_to_a','3','first_100'),('final_b','3','probe_b')]:
    pairs=[(metrics[(s,family,length,phase,measure,'correction')],metrics[(s,family,length,phase,measure,'no_correction')]) for s in sorted(seeds,key=int)]
    out['conditions'][f'{family}:{length}:{label}']={'accuracy':summarize(pairs,f"{plan['bootstrap']['seed']}:{family}:{length}:{label}:acc"), 'mixture_mean':statistics.mean(metrics[(s,family,length,phase,measure,'mixture')] for s in seeds)}
   for phase in ('2','3'):
    for window in ('first_100','remaining'):
     key=f'{family}:{length}:phase_{phase}:{window}'
     row={}
     for idx,name in enumerate(('pre_assignment_error','post_assignment_error','b_memory_mse')):
      pairs=[(diag[(s,family,length,phase,window,'correction')][idx],diag[(s,family,length,phase,window,'no_correction')][idx]) for s in sorted(seeds,key=int)]
      row[name]=summarize(pairs,f"{plan['bootstrap']['seed']}:{key}:{name}")
     margins=[mappings[(s,family,length,'correction')][1] for s in seeds]
     row['correction_alignment_margin']={'mean':statistics.mean(margins),'below_0_01':sum(x<0.01 for x in margins)}
     out['conditions'][key]=row
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
