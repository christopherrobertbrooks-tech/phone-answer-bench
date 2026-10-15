"""answer_report.py LABEL... -- re-score saved answer-brain replies against the CURRENT quiz key and scorer.

The replies in results/answers-LABEL.json never change; when the key or the scorer is corrected, this shows what
the same replies score now (the old outcome is kept as `outcome_at_run`).
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import answers  # noqa: E402

items = {i["id"]: i for i in answers.load(os.path.join(ROOT, "data", "questions.jsonl"))}
for label in sys.argv[1:]:
    path = os.path.join(ROOT, "results", f"answers-{label}.json")
    rows = json.load(open(path))
    for r in rows:
        r.setdefault("outcome_at_run", r["outcome"])
        r["outcome"] = answers.outcome(items[r["id"]], r["text"], bool(r["search"]))
    json.dump(rows, open(path, "w"), indent=1)
    print(answers.summary(label, rows))
    changed = [r for r in rows if r["outcome"] != r["outcome_at_run"]]
    for r in changed:
        print(f"    {r['id']} {r['outcome_at_run']} -> {r['outcome']}: {r['text'][:100]!r}")
