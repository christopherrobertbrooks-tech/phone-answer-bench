"""The quiz for a phone voice assistant's "answer brain": how a question is put to a model and how its reply is scored.

A voice assistant has to answer from what it knows, search the web for anything current, read a search result back,
handle follow-up questions, and never guess about the user's own life. data/questions.jsonl holds 115 hand-written
questions for exactly that; every check here is a fixed rule, not a model's opinion:

  know / howto / followup   right = no search and every key group found in the answer
  result                    the search already happened; right = every key group found (details only the result has)
  fresh                     right = it searched; answering from memory = made up
  personal                  right = it says it doesn't know, without searching

Outcomes: right, searched, broken_call, declined, asked, wrong, made_up. Only `right` is a pass; the others say what
kind of miss it was (searching a stable fact is slow but harmless, making up a score is not). `asked`/`declined` = it
asked the user something back or offered to search instead of searching. `broken_call` = a search written as text the
server could not parse (on a phone a grammar would force the call format) -- unless real prose follows the tool name,
which is scored as prose.

Spoken answers write numbers as words ("sixty-eight thousand four hundred"), so keys are matched against the reply
AND a copy with spelled-out numbers turned into digits. Curly apostrophes are straightened first.
"""
import json
import re

SYSTEM = (
    "You are the voice assistant on the user's phone. Your answer is read aloud, so answer in one to three short "
    "sentences of plain words: no lists, no markdown, no emoji. If the question needs current or local information "
    "(weather, news, scores, prices, store hours, traffic, release dates, anything that changes), call web_search "
    "instead of answering from memory. When search results are given, answer from them. If the question is about "
    "the user's own life (family, accounts, belongings, plans) and you were not told, say you don't know. Never guess."
)

TOOLS = [{"type": "function", "function": {
    "name": "web_search",
    "description": "Search the web for current or local information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to search for"}},
                   "required": ["query"]}}}]

REFUSAL = re.compile(
    r"\b(don'?t|do not|can'?t|cannot|not sure|no way|unable|no access|not able|haven'?t|wouldn'?t know|"
    r"not something i|i'?m not aware|isn'?t something i|no information)\b", re.I)

# offering to search, or saying what it would need, without doing it: a miss, but no claim was made
NEEDS = re.compile(r"\b(i need to know|i need more|i need to search|need your location|i can search|i'?ll search|"
                   r"let me search|would you like me to search|i can look (that|it) up|i can look up|i can check|"
                   r"if you (tell|give) me|what city|which city|your (zip|location|city))\b", re.I)
BROKEN_CALL = re.compile(r"^\s*(call:)?web_search\b\s*(\{.*?\})?\s*", re.S)

LONG_WORDS = 60   # a spoken answer longer than this is a lecture, reported separately (not a fail)


def load(path="data/questions.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def messages(item):
    """The conversation a model sees for one quiz item."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if item["kind"] == "followup":
        q1, a1 = item["history"]
        msgs += [{"role": "user", "content": q1}, {"role": "assistant", "content": a1}]
    msgs.append({"role": "user", "content": item["say"]})
    if item["kind"] == "result":
        call = {"id": "call0", "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps({"query": item["query"]})}}
        msgs += [{"role": "assistant", "content": "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": "call0", "content": item["result"]}]
    return msgs


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                      "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
_NUMWORD = "|".join(sorted(list(_UNITS) + list(_TENS) + list(_SCALES), key=len, reverse=True))
_RUN = re.compile(rf"\b(?:{_NUMWORD})(?:[\s-]+(?:and[\s-]+)?(?:{_NUMWORD}))*"
                  rf"(?:\s+point(?:\s+(?:{'|'.join(list(_UNITS)[:10])}))+)?\b", re.I)


def _joins(current, v):
    """Whether number word value `v` continues the number so far: "twenty" + "one", "hundred" + "twelve" do;
    "nineteen" + "sixty" (a spoken year) and "five" + "five" (a phone number) start a new chunk instead."""
    if current % 100 == 0:
        return True
    return v < 10 and current % 10 == 0 and current % 100 >= 20


def _value(run):
    whole, _, frac = run.lower().partition(" point ")
    chunks, total, current, started = [], 0, 0, False
    for w in re.split(r"[\s-]+", whole):
        if w == "zero":   # always its own digit ("five zero one" is a phone number, not 51)
            if started:
                chunks.append(total + current)
            chunks.append(0)
            total, current, started = 0, 0, False
        elif w in _UNITS or w in _TENS:
            v = _UNITS.get(w, _TENS.get(w, 0))
            if started and not _joins(current, v):
                chunks.append(total + current)
                total, current = 0, 0
            current += v
            started = True
        elif w == "hundred":
            current, started = (current or 1) * 100, True
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current, started = 0, True
    if started:
        chunks.append(total + current)
    digits = "".join(str(_UNITS[w]) for w in frac.split()) if frac else ""
    return "".join(str(c) for c in chunks) + ("." + digits if digits else "")


def digits(text):
    """`text` with spelled-out numbers turned into digits: "one hour and thirty-eight minutes" -> "1 hour and 38 minutes"."""
    return _RUN.sub(lambda m: _value(m.group(0)), text)


def keys_found(keys, text):
    both = text + " \n " + digits(text)
    return all(any(re.search(alt, both, re.I) for alt in group) for group in keys)


def outcome(item, text, searched):
    """One of right / searched / declined / wrong / made_up for a reply (`searched` = it called web_search)."""
    # curly quotes (Ministral writes "I don\u2019t know") would slip past every "don'?t"-style rule
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    kind = item["kind"]
    m = BROKEN_CALL.match(text)
    if m and not searched:
        rest = text[m.end():].strip()
        if len(rest.split()) < 4:
            return "broken_call"
        text = rest   # a tool name followed by real prose: score the prose
    asked = "?" in text   # a question back to the user anywhere in the reply ("What city are you in? I can look it up.")
    if kind == "fresh":
        if searched:
            return "right"
        if re.search(r"\d", digits(text)) and not REFUSAL.search(text):
            return "made_up"   # a specific figure without searching ("typically closes at 9:00 PM. I can check...")
        return "declined" if REFUSAL.search(text) or NEEDS.search(text) else "asked" if asked else "made_up"
    if kind == "personal":
        if searched:
            return "searched"
        return "right" if REFUSAL.search(text) else "asked" if asked else "made_up"
    if searched:
        return "searched"
    if keys_found(item["keys"], text):
        return "right"
    return "declined" if REFUSAL.search(text) or NEEDS.search(text) else "asked" if asked else "wrong"


def words(text):
    return len((text or "").split())


KINDS = ("know", "howto", "fresh", "result", "followup", "personal")


def summary(label, rows):
    """The per-kind table for saved replies (rows from results/answers-LABEL.json, outcomes already filled in)."""
    import collections
    import statistics
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["kind"]][r["outcome"]] += 1
    lines = [f"== {label}: {sum(r['outcome'] == 'right' for r in rows)}/{len(rows)} right"]
    for kind in KINDS:
        c = by[kind]
        lines.append(f"  {kind:9} {c['right']:3}/{sum(c.values()):<3} " + "  ".join(f"{k} {v}" for k, v in c.items() if k != "right"))
    spoken = [r["words"] for r in rows if r["text"]]
    lines.append(f"  words per answer: median {statistics.median(spoken):.0f}, over {LONG_WORDS}: "
                 f"{sum(w > LONG_WORDS for w in spoken)}; median seconds {statistics.median(r['seconds'] for r in rows):.2f}")
    return "\n".join(lines)
