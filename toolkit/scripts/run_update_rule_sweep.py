#!/usr/bin/env python3
"""Run and validate the frozen matched switch-rate/update-rule study."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from toolkit.coordinator import runner

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if len(sys.argv)!=4:
        raise SystemExit('usage: run_update_rule_sweep.py <protocol.json> <config.cfg> <new-output-directory>')
    protocol_path,config,output=(Path(x).resolve() for x in sys.argv[1:])
    plan=runner.read_json(protocol_path); cfg=runner.config_parse(config.read_text())
    if plan.get('schema_version')!=1 or plan['phase_lengths']!=[100,400,1000] or len(cfg['seeds'])!=30:
        raise ValueError('unsupported study definition')
    if output.exists() or output.is_symlink(): raise ValueError('output already exists')
    command=['cargo','run','--offline','--release','--manifest-path',str(ROOT/'toolkit/Cargo.toml'),
             '--bin','update_rule_sweep','--',str(config),str(output)]
    subprocess.run(command,cwd=ROOT,check=True)
    fields=['seed','family','phase_length','phase','measurement','model','correct','trials','accuracy','logloss']
    groups=defaultdict(list); keys=set(); seeds={str(x) for x in cfg['seeds']}
    with (output/'metrics.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=fields: raise ValueError('unexpected metrics columns')
        for row in reader:
            key=tuple(row[k] for k in ('seed','family','phase_length','phase','measurement','model'))
            seed,family,length,phase,measurement,model=key
            if seed not in seeds or family not in plan['families'] or int(length) not in plan['phase_lengths'] or phase not in ('1','2','3') or measurement not in ('online','first_100','probe_a','probe_b') or model not in plan['models'] or key in keys:
                raise ValueError(f'unexpected or duplicate observation: {key}')
            keys.add(key); correct=int(row['correct']); trials=int(row['trials']); accuracy=float(row['accuracy']); loss=float(row['logloss'])
            want=int(length) if measurement=='online' else (100 if measurement=='first_100' else cfg['test_tokens'])
            if trials!=want or not 0<=correct<=trials or abs(accuracy-correct/trials)>5.1e-9 or not 0<=loss<float('inf'):
                raise ValueError(f'invalid observation: {key}')
            groups[(family,length,phase,measurement,model)].append((correct/trials,loss))
    expected=len(seeds)*len(plan['families'])*len(plan['phase_lengths'])*3*4*len(plan['models'])
    if len(keys)!=expected: raise ValueError(f'incomplete metrics: {len(keys)} versus {expected}')
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        out=csv.writer(f);out.writerow(['family','phase_length','phase','measurement','model','seeds','mean_accuracy','mean_logloss'])
        for key,values in sorted(groups.items()):
            out.writerow([*key,len(values),f'{mean(v[0] for v in values):.8f}',f'{mean(v[1] for v in values):.8f}'])
    sources=[ROOT/'toolkit/src/bin/update_rule_sweep.rs',ROOT/'toolkit/src/milestone_candidates/idea1.rs',ROOT/'toolkit/src/milestone_candidates/idea2.rs',Path(__file__).resolve()]
    manifest={'revision':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),
              'working_tree_dirty':bool(subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout),
              'protocol_sha256':digest(protocol_path),'config_sha256':digest(config),
              'source_sha256':{str(x.relative_to(ROOT)):digest(x) for x in sources},
              'metrics_sha256':digest(output/'metrics.csv'),'summary_sha256':digest(output/'summary.csv')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Completed validated study:',output)

if __name__=='__main__': main()
