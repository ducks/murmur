# Optional LangGraph adapter experiment

This adapter is an experiment in orchestration, not a new model architecture. It uses LangGraph to inspect an existing Murmur campaign and its local research-memory index, optionally run at most one existing campaign iteration, and inspect the resulting state. Murmur's own campaign code still creates worktrees, invokes the configured agent, runs the experiments, saves evidence and enforces budgets.

Run a read-only preview (no model service is called):

```sh
nix-shell toolkit/adapters/langgraph/shell.nix --run 'python3 -B -m toolkit.adapters.langgraph.workflow toolkit/runs/milestone-first'
```

To run one idea in a separate ready campaign, append `--execute` and provide that campaign's path. That delegates to `campaign.run(..., max_new=1)` and may use the configured coding-agent service and incur its usual cost. Completed or failed campaigns are only inspected. Agent source code is still untrusted. The adapter disables LangChain/LangSmith tracing for this local process.

```sh
nix-shell toolkit/adapters/langgraph/shell.nix --run 'python3 -B -m unittest discover -s toolkit/adapters/langgraph/tests -v'
```

The optional Nix shell provides `langgraph` and `httpx`; the inspected local nixpkgs revision provided LangGraph 0.2.21, but the Nix package set is not pinned. It may download dependencies when first entered. The default Murmur shell and runner do not depend on LangGraph. This graph does not install a LangGraph checkpointer or implement independent crash recovery; durable records remain Murmur's campaign files. The preview and mocked execution tests establish an integration path, not scientific improvement or autonomous diagnosis. A next comparison could evaluate checkpoint/retry behavior against Murmur's existing workflow before adopting the framework more broadly.
