#!/usr/bin/env python3
"""Post-hoc slot drift summary; selects the higher-P(1) slot at end of B."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv)!=4:
        raise SystemExit('usage: summarize_memory_drift.py <evidence-directory> <family> <phase-length>')
    root=Path(sys.argv[1]);family=sys.argv[2];length=sys.argv[3]
    with (root/'states.csv').open(newline='',encoding='utf-8') as f:
        states={(r['seed'],r['phase'],r['model']):r for r in csv.DictReader(f) if r['family']==family and r['phase_length']==length}
    with (root/'metrics.csv').open(newline='',encoding='utf-8') as f:
        metrics={(r['seed'],r['model']):float(r['accuracy']) for r in csv.DictReader(f)
                 if r['family']==family and r['phase_length']==length and r['phase']=='3' and r['measurement']=='probe_b'}
    models={'mixture':'mixture','idea2_inspect':'idea2_posterior_1_25','posterior_2':'posterior_2'}
    seeds={seed for seed,phase,model in states if phase=='2' and model=='mixture'}
    if not seeds: raise ValueError('no data for requested condition')
    result={'family':family,'phase_length':int(length),'seeds':len(seeds),'models':{},
            'method':'Select the slot with higher mean P(1|x) across x=0,1 after B, then inspect the same slot after A. This heuristic is post-hoc and is not a context label.',
            'limitations':['Descriptive grouping on the same observations; no causal claim.', 'P(1|1) threshold is post-hoc and ignores online context belief.']}
    for state_model,metric_model in models.items():
        before=[];after=[];row1=[];groups=defaultdict(list)
        for seed in sorted(seeds,key=int):
            b=states[(seed,'2',state_model)]; a=states[(seed,'3',state_model)]
            slot=0 if float(b['p00'])+float(b['p01'])>=float(b['p10'])+float(b['p11']) else 1
            slotmean=lambda row:(float(row[f'p{slot}0'])+float(row[f'p{slot}1']))/2
            before.append(slotmean(b));after.append(slotmean(a)); p1=float(a[f'p{slot}1']);row1.append(p1)
            delta=metrics[(seed,metric_model)]-metrics[(seed,'mixture')]
            groups['below_0_5' if p1<0.5 else 'at_least_0_5'].append(delta)
        result['models'][state_model]={'selected_slot_mean_p1_after_b':statistics.mean(before),
            'selected_slot_mean_p1_after_a':statistics.mean(after), 'selected_slot_p1_given_1_after_a_mean':statistics.mean(row1),
            'selected_slot_p1_given_1_below_half_count':sum(x<0.5 for x in row1),
            'recall_difference_vs_mixture_by_threshold':{k:{'count':len(v),'mean':statistics.mean(v)} for k,v in sorted(groups.items())}}
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
