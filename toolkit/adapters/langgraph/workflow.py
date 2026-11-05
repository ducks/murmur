"""Optional LangGraph adapter for one bounded Murmur campaign iteration.

Preview is read-only and needs no model service. --execute delegates to the
existing campaign driver, which may invoke its configured coding-agent service.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, TypedDict

# The local adapter does not opt into hosted LangSmith tracing.
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise ImportError('LangGraph is optional. Run this adapter in toolkit/adapters/langgraph/shell.nix.') from exc
from toolkit.coordinator import campaign, memory, runner


class WorkflowState(TypedDict, total=False):
    directory: str
    mode: str
    action: str
    completed: int
    limit: int
    memory_rounds: int
    findings: int
    memory_warnings: list[dict[str, Any]]
    campaign_result: dict[str, Any] | None


def inspect(state: WorkflowState) -> dict[str, Any]:
    root = Path(state['directory']).resolve()
    config = runner.read_json(root / 'campaign.json')
    current = runner.read_json(root / 'state.json')
    if config.get('schema_version') != 1 or current.get('schema_version') != 1 or not isinstance(current.get('completed'), list):
        raise runner.PlanError('invalid campaign metadata')
    completed = len(current['completed'])
    limit = config['max_ideas']
    action = ('inspect_failure' if current.get('status') != 'ready' else
              'review_results' if completed >= limit else 'continue')
    index = memory.build_index()
    return {'directory': str(root), 'action': action, 'completed': completed, 'limit': limit,
            'memory_rounds': len(index['campaign_rounds']), 'findings': len(index['findings']),
            'memory_warnings': index['warnings']}


def route(state: WorkflowState) -> str:
    return 'run' if state['mode'] == 'execute' and state['action'] == 'continue' else 'finish'


def run_one(state: WorkflowState) -> dict[str, Any]:
    return {'campaign_result': campaign.run(state['directory'], max_new=1)}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node('inspect', inspect)
    graph.add_node('run', run_one)
    graph.add_node('reinspect', inspect)
    graph.add_edge(START, 'inspect')
    graph.add_conditional_edges('inspect', route, {'run': 'run', 'finish': END})
    graph.add_edge('run', 'reinspect')
    graph.add_edge('reinspect', END)
    return graph.compile()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='existing Murmur campaign directory')
    parser.add_argument('--execute', action='store_true', help='run at most one idea (may use a configured model service)')
    args = parser.parse_args(argv)
    try:
        result = build_graph().invoke({'directory': args.directory, 'mode': 'execute' if args.execute else 'preview'})
        print(json.dumps(result, indent=2, sort_keys=True))
    except (runner.PlanError, OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
