import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Chat with a merged Lu-3 model.")
parser.add_argument("--model", default=str(project_dir / config["output_dir"] / "merged"),
                    help="Merged model folder, or a Hugging Face model id such as Qwen/Qwen3-0.6B.")
parser.add_argument("--history", type=int, default=3, help="Exchanges of history to keep (training used up to 3).")
parser.add_argument("--greedy", action="store_true", help="Disable sampling.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16 if use_bf16 else torch.float32
).to(device).eval()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

settings = {"max_new_tokens": config["max_new_tokens"], "pad_token_id": tokenizer.pad_token_id, "streamer": streamer}
if args.greedy:
    settings.update(do_sample=False, temperature=None, top_p=None, top_k=None)
else:
    # Qwen's recommended non-thinking sampling settings.
    settings.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

print(f"Chatting with {args.model} on {device}. Commands: /reset, /quit")
history = []

while True:
    try:
        user_text = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_text:
        continue
    if user_text in ("/quit", "/exit"):
        break
    if user_text == "/reset":
        history = []
        print("(history cleared)")
        continue

    history.append({"role": "user", "content": user_text})
    history = history[-(args.history * 2 - 1):]

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": config["system_prompt"]}] + history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("Lu: ", end="", flush=True)
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, **settings)
    elapsed = time.perf_counter() - started

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": reply})
    print(f"({len(new_tokens)} tokens, {len(new_tokens) / elapsed:.1f} tok/s)")
