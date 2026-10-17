# phone-answer-bench

**Which small open model (2-4B) should answer questions on a phone?** Eight models, 115 hand-written voice-assistant
questions, a fixed-rule scorer that was itself tested before any model was scored, llama.cpp 4-bit.

**Winner: Ministral 3 3B** (Mistral, Apache-2.0, 2.15 GB at Q4_K_M) -- 112/115 in every run at its recommended
temperature, nothing made up, and the fastest of the eight. Google's Gemma 4 E4B, Qwen3.5-4B and MiniCPM5-2B tie at
107 on a greedy run; at their makers' recommended sampling they fall further behind (99-105).

| Model (maker, licence) | Q4 file | Greedy run | 3 runs, maker's sampling | Made up |
|---|---|---|---|---|
| **Ministral 3 3B** (Mistral, Apache-2.0) | 2.15 GB | **112** | **112.0** (112, 112, 112) | 0 |
| Qwen3.5-4B (Alibaba, Apache-2.0) | 2.74 GB | 107 | 104.7 | 0 |
| Gemma 4 E4B (Google, Apache-2.0) | 4.59 GB | 107 | 99.3 (89 -- 105) | 0 |
| MiniCPM5-2B (OpenBMB, Apache-2.0) | 1.56 GB | 107 | 99.7 (93 -- 104) | 0 |
| Granite 4.2 3B (IBM, Apache-2.0) | 2.24 GB | 104 | | 0 |
| Qwen3.5-2B (Alibaba, Apache-2.0) | 1.28 GB | 101 | | 3 |
| Spark-X2.5-4B (XHToken, Apache-2.0) | 2.60 GB | 96 | | 2 |
| Gemma 4 E2B (Google, Apache-2.0) | 2.84 GB | 94 | | 1 |

"Made up" is counted on the greedy run. Full tables, per question kind, are under [Results](#results).

**Update, 2026-10-17: six more models** -- Meta's MobileLLM-Pro and four MobileMoE models, and NVIDIA's Nemotron Labs
Dory 4B. None beats Ministral; the best, MobileMoE-L, scores 103. See [Six more models](#six-more-models-2026-10-17).

## The test

A phone assistant's "answer brain" gets the questions that are not commands. It has to:

| Kind | Count | Example | Right means |
|---|---|---|---|
| know | 30 | "Who wrote Frankenstein?" | the answer, without searching |
| howto | 20 | "How do I jump start a car?" | every key step |
| fresh | 20 | "Who won the Bears game last night?" | it calls `web_search` (answering from memory = **made up**) |
| result | 20 | the same kind of question, with a search result already supplied | the details only the result contains (a 2026 forecast, a local phone number) |
| followup | 15 | "Who painted the Mona Lisa?" ... "What else did he paint?" | the answer, using the earlier turn |
| personal | 10 | "Where did I park my car?" | "I don't know", without searching or guessing |

Every model gets the same short system prompt (answer in one to three spoken sentences; search for anything current;
read search results back; never guess about the user's life) and one tool, `web_search(query)`. Thinking is off
(`reasoning_effort: none`, `enable_thinking: false`). See [`answers.py`](answers.py).

**The scorer is rules, not a judge model.** Each question has key groups (every group must match); `fresh` and
`personal` are scored on behaviour. Outcomes other than `right` say what kind of miss it was: `searched` (searched a
stable fact -- slow but harmless), `declined` / `asked` (offered to search or asked "what city are you in?" instead of
searching), `broken_call` (a search written as text the server could not parse), `wrong`, and `made_up` (a current
fact stated without searching -- the one a voice assistant must never do).

**The scorer was tested before it scored anything** ([`tests/test_answers.py`](tests/test_answers.py)): every right
answer passes, empty / "Maybe later." / "I'm not sure." fail, and each question's right answer given to every other
question passes less than 1% of the time.

**It was still wrong six times on the first evening, and every fix came from reading misses, not from scores:**

1. Two keys accepted "I'm **not** sure" as a correct "no" (snow forecast, post office). Found by the tests.
2. Spoken answers spell numbers out ("one hour and thirty-eight minutes"); keys now also match a digits copy
   (years, phone numbers and decimals handled).
3. "What city are you in?" was scored as *made up*; asking back is now its own outcome.
4. Ministral writes `don’t` with a curly apostrophe, which slipped past every "don't"-style rule: its ten correct
   "I don't know your sister's birthday" answers were scored as misses.
5. Over-strict keys: "No snow is expected" (correct, the key wanted the forecast details too), "The Bears won"
   (correct, the key wanted the score), "186,282 miles per second" (correct, the key wanted 186,000).
6. "Home Depot typically closes at 9:00 PM -- I can check" was scored as *offered to check*; a specific figure stated
   without searching is now *made up* even when an offer follows.

All runs below are scored on the final rules ([`scripts/summarize.py`](scripts/summarize.py) re-scores the saved
replies; the replies themselves never change).

## Results

<!-- tables: scripts/summarize.py -->

### One greedy run per model

| Model | Right /115 | know /30 | howto /20 | fresh /20 | result /20 | followup /15 | personal /10 | Made up | Wrong |
|---|---|---|---|---|---|---|---|---|---|
| ministral3-3b | **112** | 30 | 20 | 20 | 19 | 13 | 10 | 0 | 1 |
| qwen35-4b | **107** | 30 | 19 | 19 | 20 | 9 | 10 | 0 | 1 |
| minicpm5-2b | **107** | 30 | 15 | 18 | 20 | 14 | 10 | 0 | 5 |
| gemma4-e4b | **107** | 30 | 18 | 19 | 20 | 10 | 10 | 0 | 2 |
| granite42-3b | **104** | 30 | 19 | 15 | 20 | 10 | 10 | 0 | 3 |
| qwen35-2b | **101** | 30 | 18 | 10 | 20 | 13 | 10 | 3 | 4 |
| spark-x25-4b | **96** | 30 | 16 | 7 | 20 | 13 | 10 | 2 | 6 |
| gemma4-e2b | **94** | 28 | 17 | 9 | 20 | 10 | 10 | 1 | 6 |

### Three runs at each maker's recommended sampling

| Model | Runs | Mean | Made up (3 runs) |
|---|---|---|---|
| ministral3-3b | 112, 112, 112 | **112.0** | 0 |
| qwen35-4b | 104, 105, 105 | **104.7** | 1 |
| minicpm5-2b | 93, 104, 102 | **99.7** | 1 |
| gemma4-e4b | 89, 105, 104 | **99.3** | 0 |

### Temperature sweep

| Model | Temperature | Runs | Mean | Made up (3 runs) |
|---|---|---|---|---|
| ministral3-3b | 0 | 112, 112, 112 | **112.0** | 0 |
| ministral3-3b | 0.1 | 112, 112, 112 | **112.0** | 0 |
| ministral3-3b | 0.3 | 111, 112, 112 | **111.7** | 1 |
| ministral3-3b | 0.5 | 112, 109, 109 | **110.0** | 2 |
| ministral3-3b | 0.7 | 109, 112, 110 | **110.3** | 2 |
| ministral3-3b | 1 | 111, 80, 108 | **99.7** | 1 |
| ministral3-3b | 1.5 | 11, 71, 56 | **46.0** | 32 |


### What the numbers say

- **Ministral 3 3B is the most reliable answerer here.** 30/30 knowledge, 20/20 how-to, 20/20 searches when it should,
  10/10 on personal questions. Its misses: one answer that named the leader but not the score, and two follow-ups it
  searched instead of answering.
- **A maker's own benchmark table is not a test.** Spark-X2.5-4B's model card shows it far ahead of Qwen3.5-4B and
  Gemma 4 E4B; here it was 7th: it asked for the user's location instead of searching 11 times out of 20, and told
  users to connect the red (positive) jump-start clamp to the dead battery's *negative* terminal.
- **Temperature matters more than the model card says.** At their makers' recommended temperature 1.0, Gemma 4 E4B
  and MiniCPM5-2B each had one bad run (89 against 104-105 for Gemma, 93 against 102-104 for MiniCPM); Gemma leaked its reasoning into spoken answers ("The
  user is asking for tomorrow's weather. I will use the current search results...") and named "Burt Alerick" as the
  second man on the moon. Ministral is flat from 0 to 0.3 and starts making things up from 0.3 (1-2 per three runs at
  0.3-0.7, none at 0-0.1); with no top-p/top-k trimming, one run at 1.0 collapsed into mixed-language gibberish
  ("Henrietta, Меркурий is the closest planet to the Sun") and at 1.5 all three runs fell apart (11, 56 and 71/115).
  For a voice assistant: run it cold.
- **Small models garble numbers.** Gemma 4 E2B: "a leap year has thirty-sixteen days", "sixty minutes in a day";
  Qwen3.5-2B: spaghetti for "2 to 3 minutes". Knowledge questions are not where the 2B models lose; numbers and
  current events are.
- **Knowing when to search is the real split.** Every model scored 28-30/30 on stable facts and 19-20/20 reading a
  supplied search result. The spread is in `fresh` (7-20/20): whether the model searches, asks for a location, or
  answers from stale memory.

## Six more models (2026-10-17)

Meta's on-device models (MobileLLM-Pro, and MobileMoE at three sizes plus the 4-bit QAT version of the largest) and
NVIDIA's Nemotron Labs Dory 1.0 4B, through the same 115 questions, greedy, one run each.

| Model (maker, licence) | Run as | Right /115 | know | howto | fresh | result | followup | personal | Made up |
|---|---|---|---|---|---|---|---|---|---|
| MobileMoE-L (Meta, FAIR Noncommercial) | transformers, fp16 | **103** | 30 | 18 | 15 | 18 | 14 | 8 | 1 |
| Nemotron Labs Dory 1.0 4B (NVIDIA, OpenMDW-1.1) | llama.cpp PR build, NVIDIA's Q5_K_M | **101** | 30 | 19 | 8 | 20 | 14 | 10 | 0 |
| MobileLLM-Pro (Meta, FAIR Noncommercial) | transformers, fp16 | **101** | 24 | 17 | 20 | 20 | 14 | 6 | 0 |
| MobileMoE-L-QAT (Meta, FAIR Noncommercial) | transformers, its INT4 weights | **87** | 26 | 5 | 13 | 19 | 14 | 10 | 0 |
| MobileMoE-S (Meta, FAIR Noncommercial) | transformers, fp16 | **76** | 29 | 16 | 4 | 12 | 11 | 4 | 15 |
| MobileMoE-M (Meta, FAIR Noncommercial) | transformers, fp16 | **69** | 18 | 0 | 20 | 17 | 13 | 1 | 0 |

What they did:

- **The MobileMoE sizes fail in opposite directions.** S answers current questions from memory (15 made up:
  "Bitcoin's price is about $60,000", "the Cubs game is currently at 3-2"). M searches almost everything, including
  all 20 how-to questions ("how do I hard boil an egg?") and 9 of 10 personal ones. L is the most balanced (30/30
  knowledge) but four times said "I don't have real-time data" instead of searching, and stated "Home Depot closes
  at 9:00 PM today" without searching.
- **Quantization-aware training cost L 16 points.** The INT4 QAT version (2.75 GB) searched 15 of 20 how-to
  questions and answered "I don't know" to seven questions it should have searched; it was perfect on personal ones.
- **MobileLLM-Pro (1B) made nothing up** and was 20/20 on searching and on reading results, but searched five stable
  facts and four personal questions, and said "86,400 minutes in a day" and "four fluid ounces" in a quart.
- **Dory 4B made nothing up and was perfect on knowledge, results and personal questions, but would not search:** 12
  of the 20 current questions got "I don't know your location" or "I don't have real-time data" instead of a search,
  the same habit that put Spark-X 7th above. It is also the wordiest here (median 24 words; 7 answers over 60), and
  the only one of the six whose licence (OpenMDW-1.1, permissive) allows commercial use.

How they were run, and why it differs from the eight above:

- **Meta's five could not run in llama.cpp** (custom architectures, not supported upstream as of this date), so
  [`scripts/hf_server.py`](scripts/hf_server.py) serves them through Hugging Face transformers 4.57.1 (their code
  predates transformers 5) on the V100, in fp16 -- full precision, which flatters them against the 4-bit GGUFs above;
  MobileMoE-L-QAT runs from its own INT4 weights. Their chat templates have no tool-call format, so the tool is
  described in one added sentence of the system prompt (reply `web_search("...")` to search), and a reply in that
  form, or JSON naming the tool, becomes a real tool call ([`tests/test_hf_server.py`](tests/test_hf_server.py)).
  MobileLLM-Pro twice searched for the example wording itself on personal questions (a miss either way).
- **Dory ran in llama.cpp** built from NVIDIA's open pull request
  ([ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306), commit `43e865ec0`, not merged as of
  this date), using NVIDIA's own Q5_K_M GGUF -- one step above the Q4_K_M used for the eight: the pull request's
  converter expects a different config layout than the released checkpoint, so a Q4_K_M could not be made from the
  released weights. On the V100 most of its work ran on the CPU (GPU at 3%, about 5.7 s per answer); speed is not
  what this bench measures.
- Replies are in [`results/oct17/`](results/oct17/); the scorer is unchanged.

## Setup

- One Tesla V100 32 GB (only the scoring runs on a GPU; the model choice is for a phone's CPU/GPU later),
  llama.cpp `5e03bdd` (`llama-server --jinja -ngl 99 -fa on -c 8192 -np 1 --cache-ram 0`), CUDA 12.9.
- Quantisation: Q4_K_M for all but the Gemma 4 E-models, which ggml-org publishes as Q4_0. Files and sources:

| Model | GGUF repository | Base model licence |
|---|---|---|
| Ministral 3 3B Instruct 2512 | `mistralai/Ministral-3-3B-Instruct-2512-GGUF` | Apache-2.0 |
| Qwen3.5-4B / 2B | `unsloth/Qwen3.5-4B-GGUF`, `unsloth/Qwen3.5-2B-GGUF` (Qwen publishes no GGUF) | Apache-2.0 |
| Gemma 4 E4B / E2B | `ggml-org/gemma-4-E4B-it-GGUF`, `ggml-org/gemma-4-E2B-it-GGUF` | Apache-2.0 |
| MiniCPM5-2B | `openbmb/MiniCPM5-2B-GGUF` | Apache-2.0 |
| Granite 4.2 3B | `ibm-granite/granite-4.2-3b-GGUF` | Apache-2.0 |
| Spark-X2.5-4B | `XHToken/Spark-X2.5-4B-GGUF` | Apache-2.0 |

- Not tested, because their licences add limits a commercial phone app would have to plan around: Liquid
  LFM2.5-2.6B (LFM Open License: commercial use only under $10M annual revenue) and NVIDIA Nemotron-3-Nano-4B (NVIDIA
  licence with conditions). Community distills of other models were skipped for unclear data provenance.
- Makers' recommended sampling used for the three-run test: Ministral temperature 0.1; Qwen3.5-4B non-thinking
  0.7 / top-p 0.8 / top-k 20 / presence penalty 1.5; Gemma 4 1.0 / 0.95 / 64; MiniCPM5 1.0 / 0.95. The temperature
  sweep changes temperature only (top-p 1, top-k off, min-p 0).

## Rerun it

```sh
python3 -m unittest discover -s tests          # the scorer can fail before it scores anything
LLAMA_SERVER=/path/to/llama-server scripts/run_model.sh mymodel model-Q4_K_M.gguf
LLAMA_SERVER=/path/to/llama-server scripts/run_model.sh mymodel-s1 model.gguf --sample 0.7,0.8,20,0.0,1.5 --seed 1
PYTHON=/path/to/venv/bin/python scripts/run_hf.sh mymodel /path/to/hf-model   # a model llama.cpp can't run (transformers)
python3 scripts/answer_report.py mymodel        # re-score saved replies after changing the key
python3 scripts/summarize.py                    # the tables above, from results/
```

Python 3 standard library only. Replies are saved with timings in `results/`.

## Limits

- 115 questions, English, US-flavoured (Chicago teams, Fahrenheit), one machine. A 3-point gap is noise; the 5-7
  point gaps that held across three runs are not.
- The quiz checks that a model *asks* for a search and *reads* a supplied result; it does not run a real search.
- Phone speed is not measured here: these are quality numbers. The same GGUF files run in llama.cpp on Android.
- Now that the questions are public, future models may be trained on them; treat later scores with that in mind.

**AI assistance:** the questions, scorer, runs and this write-up were done with an AI coding assistant (Claude, by
Anthropic). A person directed the work and reviewed the results. Every number here comes from a JSON file under
`results/`.

Code and questions: MIT (see [LICENSE](LICENSE)). The saved model replies are outputs of the models listed above.
