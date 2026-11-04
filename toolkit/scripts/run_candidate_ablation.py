#!/usr/bin/env python3
"""Run and summarize the matched prior-ablation suite in a new output directory."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: run_candidate_ablation.py <config> <new-output-directory>')
    config = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f'output already exists: {output}')
    sources = [ROOT / 'toolkit/src/bin/candidate_ablation.rs',
               *(ROOT / 'toolkit/src/candidate_transfer' / f'{name}.rs' for name in ('posterior', 'predictive', 'neutral')),
               ROOT / 'toolkit/scripts/candidate_transfer_summary.py', ROOT / 'toolkit/scripts/candidate_ablation_report.py', Path(__file__).resolve()]
    command = ['cargo', 'run', '--offline', '--release', '--manifest-path', str(ROOT / 'toolkit/Cargo.toml'),
               '--bin', 'candidate_ablation', '--', str(config), str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    shutil.copyfile(config, output / 'config.cfg')
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
        subprocess.run([sys.executable, str(ROOT / 'toolkit/scripts/candidate_transfer_summary.py'), str(output / 'metrics.csv')],
                       cwd=ROOT, stdout=f, check=True)
    with (output / 'report.md').open('w', encoding='utf-8') as f:
        subprocess.run([sys.executable, str(ROOT / 'toolkit/scripts/candidate_ablation_report.py'), str(output / 'metrics.csv')],
                       cwd=ROOT, stdout=f, check=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True, capture_output=True, check=True).stdout
    manifest = {'revision': revision, 'working_tree_dirty': bool(dirty), 'config_sha256': digest(config),
                'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sources},
                'metrics_sha256': digest(output / 'metrics.csv'), 'summary_sha256': digest(output / 'summary.csv'), 'report_sha256': digest(output / 'report.md')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Completed: {output} (metrics.csv, summary.csv, report.md, config.cfg, manifest.json)')

if __name__ == '__main__':
    main()
