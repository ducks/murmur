from pathlib import Path
from unittest.mock import patch
import json
from toolkit.coordinator import campaign, runner

root=Path('toolkit/runs/recovery-smoke-01').resolve()
campaign.init(root, 'exercise bounded recovery with synthetic counts', 2, 60, 180,
              protocol='milestone', diagnose_failures=False, seed_candidate='aad7c48287191eef0762906e47ebcb67d03e02c2', max_recoveries=1)
fields=campaign.MILESTONE_FIELDS
milestone={x:'Synthetic recovery smoke test only.' for x in fields}
milestone.update(scope='test_only',primary_measure='smoke',candidate_label='candidate',control_label='control')
def planner(config,worktree,round_dir,history,research):
    runner.write_json(round_dir/'milestone.json',milestone)
    return milestone

def agent(config,worktree,round_dir,history):
    test=worktree/'toolkit/tests/agent_generated.rs'
    if round_dir.name=='idea-001':
        test.write_text('#[test]\nfn deliberate_failure() { assert_eq!(1, 2); }\n')
    else:
        assert history[-1]['recovery_eligible'] is True
        assert history[-1]['recovery_scheduled'] is True
        test.write_text(r'''#[test]
fn synthetic_recovery_smoke() {
    assert_eq!(1 + 1, 2);
    for seed in 0..100 {
        println!("MURMUR_PAIRED {{\"measure\":\"smoke\",\"candidate\":\"candidate\",\"control\":\"control\",\"seed\":{},\"candidate_correct\":1,\"control_correct\":1,\"trials\":2}}", seed);
    }
    println!("MURMUR_FOLLOWUP {{\"measure\":\"smoke\",\"candidate\":\"candidate\",\"control\":\"control\",\"candidate_correct\":100,\"control_correct\":100,\"trials\":200}}");
}
''')
    return {'question':'Synthetic recovery smoke test','hypothesis':'A retry follows an ordinary failed test.',
            'falsification':'No recovery is scheduled.','approach':'Intentional failure then scripted test.',
            'limitations':'Synthetic workflow check only; not a model experiment.'}

with patch.object(campaign,'plan_milestone',side_effect=planner):
    result=campaign.run(root,agent=agent)
print(json.dumps(result,indent=2,sort_keys=True))
