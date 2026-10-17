"""hf_server.py MODEL_DIR [--port 8490] -- an OpenAI-style /v1/chat/completions for a model llama.cpp can't run yet,
so answer_quiz.py can ask it exactly what it asks llama-server.

Added 2026-10-17 for Meta's MobileLLM-Pro and MobileMoE (custom architectures, trust_remote_code; no llama.cpp
support, upstream included). Differences from the llama-server runs, said in the results:
- full precision (fp16 on the V100; bf16 is ~9x slower on Volta), not 4-bit;
- their chat templates have no tool-call format, so the tool is described in one added sentence of the system prompt
  (reply `web_search("...")` to search), and a reply in that form -- or as JSON naming web_search -- becomes a real
  tool call, as llama-server does for templates without one. A search result is passed back as a user turn.
Greedy only (temperature 0), max_tokens as asked; stops at the model's end-of-turn.
"""
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOL_NOTE = ("\n\nYou have one tool, web_search. To search the web, reply with only this one line and nothing else: "
             "web_search(\"what to search for\"). The results come back to you in the next message.")

_CALL = re.compile(r"web_search\s*\(\s*(?:query\s*=\s*)?([\"'])(.+?)\1\s*\)", re.S | re.I)   # MobileMoE-M wrote "Web_search(...)"
_JSON = re.compile(r"\{.*\}", re.S)


def find_call(text):
    """The search a reply asks for (web_search("x"), web_search(query='x'), or JSON naming web_search), else None."""
    m = _CALL.search(text or "")
    if m:
        return m.group(2).strip()
    j = _JSON.search(text or "")
    if j:
        try:
            o = json.loads(j.group(0))
        except ValueError:
            return None
        o = o.get("tool_call", o) if isinstance(o, dict) else None
        if isinstance(o, dict) and o.get("name") == "web_search":
            args = o.get("arguments", o.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            q = args.get("query") if isinstance(args, dict) else None
            return q.strip() if isinstance(q, str) and q.strip() else None
    return None


def plain(messages):
    """The bench's messages in a form these templates take: the tool described in the system turn, a tool call as the
    assistant saying it, a tool result as a user turn."""
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append({"role": "system", "content": m["content"] + TOOL_NOTE})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            q = json.loads(m["tool_calls"][0]["function"]["arguments"]).get("query", "")
            out.append({"role": "assistant", "content": f'web_search("{q}")'})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": "Search results:\n" + m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def reply(text, prompt_n, predicted_n, seconds):
    """The OpenAI-style answer: a tool call when the reply asks for a search, else the words."""
    q = find_call(text)
    msg = {"role": "assistant", "content": text.strip()}
    if q is not None:
        msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call0", "type": "function", "function": {
            "name": "web_search", "arguments": json.dumps({"query": q})}}]}
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if q is not None else "stop"}],
            "timings": {"prompt_n": prompt_n, "predicted_n": predicted_n,
                        "predicted_per_second": round(predicted_n / seconds, 1) if seconds > 0 else None}}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = sys.argv[1]
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8490
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # straight onto the GPU (the gateway has 16 GB of RAM), fp16 (Volta)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.float16,
                                                 device_map={"": 0}, low_cpu_mem_usage=True).eval()
    eot = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot|>")} if isinstance(i, int) and i >= 0]

    class H(BaseHTTPRequestHandler):
        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self.send(200, {"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            ids = tok.apply_chat_template(plain(body["messages"]), add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=int(body.get("max_tokens", 400)), do_sample=False,
                                     eos_token_id=eot, pad_token_id=eot[0])
            new = out[0][ids["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            self.send(200, reply(text, int(ids["input_ids"].shape[1]), int(len(new)), time.perf_counter() - t0))

        def log_message(self, *a):
            pass

    print(f"hf_server: {path} on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    main()
