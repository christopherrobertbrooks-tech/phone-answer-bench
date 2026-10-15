"""answer_quiz.py BASE_URL LABEL [--think] [--sample T,TOP_P,TOP_K,MIN_P,PRESENCE --seed N]
-- put the 115 answer-brain questions to an OpenAI-compatible server.

BASE_URL e.g. http://127.0.0.1:8490/v1 (llama-server --jinja). Greedy by default, one reply per item, the
web_search tool offered on every item (answers.TOOLS). Thinking is off unless --think (the phone wants short,
fast replies). --sample uses the maker's recommended sampling instead of greedy (the 3-run test, 2026-10-14),
with --seed so each run is different but repeatable.
Writes results/answers-LABEL.json (every reply, its outcome, timings) and prints a per-kind table.
"""
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import answers  # noqa: E402

base, label = sys.argv[1].rstrip("/"), sys.argv[2]
think = "--think" in sys.argv
sample = sys.argv[sys.argv.index("--sample") + 1] if "--sample" in sys.argv else None
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
out_path = os.path.join(ROOT, "results", f"answers-{label}.json")
rows = []
for item in answers.load(os.path.join(ROOT, "data", "questions.jsonl")):
    body = {"model": label, "messages": answers.messages(item), "tools": answers.TOOLS, "temperature": 0,
            "max_tokens": 1500 if think else 400, "stream": False,
            "chat_template_kwargs": {"enable_thinking": think}}
    if not think:
        body["reasoning_effort"] = "none"
    if sample:
        t, top_p, top_k, min_p, presence = (float(x) for x in sample.split(","))
        body.update(temperature=t, top_p=top_p, top_k=int(top_k), min_p=min_p, presence_penalty=presence, seed=seed)
    t0 = time.perf_counter()
    d = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                                                headers={"Content-Type": "application/json"}), timeout=600))
    msg = d["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    text = msg.get("content") or ""
    searched = any(c.get("function", {}).get("name") == "web_search" for c in calls)
    t = d.get("timings", {})
    rows.append({"id": item["id"], "kind": item["kind"], "say": item["say"], "text": text,
                 "search": [c["function"].get("arguments") for c in calls], "outcome": answers.outcome(item, text, searched),
                 "words": answers.words(text), "seconds": round(time.perf_counter() - t0, 2),
                 "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
                 "decode_tok_s": t.get("predicted_per_second"), "reasoning_chars": len(msg.get("reasoning_content") or "")})
    r = rows[-1]
    print(f"{r['id']} {r['kind']:8} {r['outcome']:8} {r['say'][:45]!r:48} -> {(text or str(r['search']))[:90]!r}", flush=True)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(rows, open(out_path, "w"), indent=1)

print("\n" + answers.summary(label, rows))
