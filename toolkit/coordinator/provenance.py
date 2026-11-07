"""Record the exact Git ancestry and file changes used by a campaign round."""
import hashlib
from pathlib import Path
import subprocess
from . import runner

ALLOWED_SEED = {'toolkit/src/agent_candidate.rs', 'toolkit/tests/agent_generated.rs'}
ALLOWED_GENERATED = {'toolkit/configs/agent/dev.cfg', 'toolkit/configs/agent/evaluation.cfg', 'toolkit/plans/agent-candidate.json'}


def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=runner.ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise runner.PlanError(f'provenance Git operation failed: {args[0]}') from exc


def exact_commit(value):
    if not isinstance(value, str) or len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
        raise runner.PlanError('provenance requires full hexadecimal commit IDs')
    if git('rev-parse', '--verify', f'{value}^{{commit}}').decode().strip() != value:
        raise runner.PlanError('provenance commit mismatch')
    return value


def diff(start, end):
    paths = git('diff', '--no-renames', '--name-only', '-z', start, end).split(b'\0')
    names = sorted(p.decode('utf-8') for p in paths if p)
    patch = git('diff', '--no-renames', '--binary', '--full-index', start, end)
    return names, patch


def blobs(commit):
    return {path: hashlib.sha256(git('show', f'{commit}:{path}')).hexdigest() for path in sorted(ALLOWED_SEED)}


def record(config, index, starting_commit, candidate_commit, directory):
    """Save paths and patches for both the seed transplant and the agent's actual delta."""
    campaign_base = exact_commit(config['base_commit'])
    start = exact_commit(starting_commit); candidate = exact_commit(candidate_commit)
    parent = git('rev-parse', f'{candidate}^').decode().strip()
    if parent != start:
        raise runner.PlanError('candidate parent does not match recorded starting commit')
    seed_source = config.get('seed_candidate_commit') if index == 1 else None
    seed_info = None
    if seed_source:
        exact_commit(seed_source)
        if start != campaign_base and git('rev-parse', f'{start}^').decode().strip() != campaign_base:
            raise runner.PlanError('seed transplant parent does not match campaign base')
        seed_names, seed_patch = diff(campaign_base, start)
        if set(seed_names) - ALLOWED_SEED or blobs(seed_source) != blobs(start):
            raise runner.PlanError('seed transplant differs from recorded source or changes other files')
        (Path(directory) / 'seed-transplant.patch').write_bytes(seed_patch)
        seed_info = {'source_commit': seed_source, 'transplant_base': campaign_base, 'transplant_commit': start,
                     'changed_paths': seed_names, 'patch_sha256': hashlib.sha256(seed_patch).hexdigest(),
                     'candidate_and_test_sha256': blobs(start)}
    agent_names, agent_patch = diff(start, candidate)
    if not agent_names or set(agent_names) - ALLOWED_SEED - ALLOWED_GENERATED:
        raise runner.PlanError('candidate delta has missing or unexpected paths')
    (Path(directory) / 'candidate-changes.patch').write_bytes(agent_patch)
    result = {'schema_version': 1, 'campaign_base_commit': campaign_base, 'round_start_commit': start,
              'candidate_commit': candidate, 'candidate_parent_commit': parent, 'seed_transplant': seed_info,
              'candidate_delta': {'from_commit': start, 'to_commit': candidate, 'changed_paths': agent_names,
                                  'patch_sha256': hashlib.sha256(agent_patch).hexdigest(),
                                  'candidate_and_test_sha256': blobs(candidate)},
              'qualification': 'Git provenance and hashes identify recorded changes; they do not establish experimental validity or safe code.'}
    runner.write_json(Path(directory) / 'provenance.json', result)
    return result
