"""summarize.py -- the README's tables, built from the saved replies in results/ (re-scored on the current rules).

results/greedy/LABEL.json            one greedy run per model
results/makers-settings/LABEL-sN.json three runs at the model maker's recommended sampling (seeds 1-3)
results/temperature/LABEL-tT-sN.json  the temperature sweep (temperature only; top_p 1, top_k off, min_p 0)
"""
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import answers  # noqa: E402

ITEMS = {i["id"]: i for i in answers.load(os.path.join(ROOT, "data", "questions.jsonl"))}


def score(path):
    rows = json.load(open(path))
    c = collections.Counter(answers.outcome(ITEMS[r["id"]], r["text"], bool(r["search"])) for r in rows)
    return c


def kinds(path):
    rows = json.load(open(path))
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["kind"]][answers.outcome(ITEMS[r["id"]], r["text"], bool(r["search"]))] += 1
    return by


def name(path):
    return os.path.basename(path)[:-5]


print("## One greedy run per model\n")
print("| Model | Right /115 | know /30 | howto /20 | fresh /20 | result /20 | followup /15 | personal /10 | Made up | Wrong |")
print("|---|---|---|---|---|---|---|---|---|---|")
rows = []
for p in glob.glob(os.path.join(ROOT, "results", "greedy", "*.json")):
    c, k = score(p), kinds(p)
    rows.append((c["right"], name(p), c, k))
for right, label, c, k in sorted(rows, reverse=True):
    print(f"| {label} | **{right}** | " + " | ".join(str(k[x]["right"]) for x in answers.KINDS) + f" | {c['made_up']} | {c['wrong']} |")

print("\n## Three runs at each maker's recommended sampling\n")
print("| Model | Runs | Mean | Made up (3 runs) |")
print("|---|---|---|---|")
runs = collections.defaultdict(list)
for p in sorted(glob.glob(os.path.join(ROOT, "results", "makers-settings", "*.json"))):
    runs[re.sub(r"-s\d$", "", name(p))].append(score(p))
for label, cs in sorted(runs.items(), key=lambda kv: -sum(c["right"] for c in kv[1])):
    rights = [c["right"] for c in cs]
    print(f"| {label} | {', '.join(map(str, rights))} | **{sum(rights) / len(rights):.1f}** | {sum(c['made_up'] for c in cs)} |")

print("\n## Temperature sweep\n")
print("| Model | Temperature | Runs | Mean | Made up (3 runs) |")
print("|---|---|---|---|---|")
sweep = collections.defaultdict(list)
for p in glob.glob(os.path.join(ROOT, "results", "temperature", "*.json")):
    m = re.match(r"(.*)-t([\d.]+)-s\d$", name(p))
    sweep[(m.group(1), float(m.group(2)))].append(score(p))
for (label, t), cs in sorted(sweep.items()):
    rights = [c["right"] for c in cs]
    print(f"| {label} | {t:g} | {', '.join(map(str, rights))} | **{sum(rights) / len(rights):.1f}** | {sum(c['made_up'] for c in cs)} |")
