#!/usr/bin/env python3
"""Validate and summarize paired slot-restoration intervention outcomes."""
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv)!=2: raise SystemExit('usage: summarize_memory_intervention.py <metrics.csv>')
    data={}; seeds=set(); families={'uneven_bias','shifted_bias'}; arms={'sham','restore_selected','restore_other','mixture'}
    with Path(sys.argv[1]).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['seed','family','arm','probe','correct','trials','accuracy','selected_slot']: raise ValueError('unexpected columns')
        for row in reader:
            key=tuple(row[x] for x in ('seed','family','arm','probe')); seed,family,arm,probe=key
            if key in data or family not in families or arm not in arms or probe not in ('a','b') or row['selected_slot'] not in ('0','1'): raise ValueError(f'unexpected row {key}')
            correct,trials=int(row['correct']),int(row['trials']);acc=float(row['accuracy'])
            if trials!=1000 or not 0<=correct<=trials or not math.isfinite(acc) or abs(acc-correct/trials)>5.1e-9: raise ValueError(f'invalid accuracy {key}')
            seeds.add(seed);data[key]=correct/trials
    if len(seeds)!=30 or len(data)!=30*len(families)*len(arms)*2: raise ValueError('incomplete evidence')
    result={'seed_count':len(seeds),'bootstrap':'10,000 paired percentile resamples, random seed 20261105, descriptive, unadjusted', 'conditions':{}}
    for family in sorted(families):
        for probe in ('a','b'):
            baseline=[data[(s,family,'sham',probe)] for s in sorted(seeds,key=int)]
            entries={}
            for arm in sorted(arms):
                vals=[data[(s,family,arm,probe)] for s in sorted(seeds,key=int)]
                diff=[a-b for a,b in zip(vals,baseline)]
                rng=random.Random(20261105);boots=sorted(statistics.mean(rng.choice(diff) for _ in diff) for _ in range(10000))
                entries[arm]={'mean_accuracy':statistics.mean(vals),'mean_difference_vs_sham':statistics.mean(diff),
                    'median_difference_vs_sham':statistics.median(diff),'wins_losses_ties':[sum(x>0 for x in diff),sum(x<0 for x in diff),sum(x==0 for x in diff)],
                    'bootstrap_95_percentile':[boots[250],boots[9750]]}
            result['conditions'][f'{family}:{probe}']=entries
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
