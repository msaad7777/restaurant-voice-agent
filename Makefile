# One command for the checks CI runs. Needs uv; no models, no network after `uv sync`.
.PHONY: check test eval

check: test eval

test:
	uv sync --frozen
	uv run pytest -q

# Rule baseline on the scripted calls: deterministic, no models. Fails below 90% task completion.
eval:
	uv run python eval/run_eval.py text --agent rules --name ci_rules > /dev/null
	uv run python -c "import json; s=json.load(open('eval/results/ci_rules.json'))['summary']; print('task_completion', s['task_completion']); assert s['task_completion'] >= 0.9"
