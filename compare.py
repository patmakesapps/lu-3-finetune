import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety import SafetyFilter

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Compare base Qwen3 and the Lu-3 adapter on held-out prompts.")
parser.add_argument("--adapter", default=str(project_dir / config["output_dir"] / "final"))
parser.add_argument("--prompts", default=str(project_dir / "data" / "eval_prompts.jsonl"))
parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts.")
parser.add_argument("--greedy", action="store_true", help="Disable sampling.")
parser.add_argument("--safety", action="store_true", help="Also show Lu-3 behind the blocklist filter (safety.py).")
args = parser.parse_args()

safety = SafetyFilter.load() if args.safety else None

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

base_model = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16 if use_bf16 else torch.float32)
model = PeftModel.from_pretrained(model, args.adapter).to(device).eval()

with Path(args.prompts).open(encoding="utf-8") as file:
    prompts = [json.loads(line) for line in file if line.strip()]

if args.limit:
    prompts = prompts[: args.limit]


def generate(messages):
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": config["system_prompt"]}] + messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    settings = {
        "max_new_tokens": config["max_new_tokens"],
        "repetition_penalty": config["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.greedy:
        settings.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        # Qwen's recommended non-thinking sampling settings.
        settings.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

    torch.manual_seed(config["seed"])
    with torch.no_grad():
        output = model.generate(**inputs, **settings)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


results = []
for number, prompt in enumerate(prompts, start=1):
    with model.disable_adapter():
        base_reply = generate(prompt["messages"])
    lu_reply = generate(prompt["messages"])
    result = {**prompt, "base": base_reply, "lu3": lu_reply}
    if safety:
        masked = [{**m, "content": safety.mask(m["content"])[0]} if m["role"] == "user" else m for m in prompt["messages"]]
        filtered = generate(masked)
        result["lu3_filtered"] = filtered if safety.is_clean(filtered) else f"(replaced) {safety.safe_reply()}"
    results.append(result)
    print(f"[{number}/{len(prompts)}] {prompt['category']}: {prompt['messages'][-1]['content']}")

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
report_dir = project_dir / "outputs" / "compare"
report_dir.mkdir(parents=True, exist_ok=True)

with (report_dir / f"compare-{stamp}.jsonl").open("w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

lines = [f"# Base vs Lu-3 ({stamp})", "", f"Base model: `{base_model}`  ", f"Adapter: `{args.adapter}`  ", f"Prompts: `{args.prompts}`", ""]
for number, result in enumerate(results, start=1):
    lines.append(f"## {number}. {result['category']}")
    lines.append("")
    for message in result["messages"]:
        speaker = "User" if message["role"] == "user" else "Lu (history)"
        lines.append(f"**{speaker}:** {message['content']}  ")
    lines += ["", f"**Base:** {result['base']}", "", f"**Lu-3:** {result['lu3']}", ""]
    if "lu3_filtered" in result:
        lines += [f"**Lu-3 + filter:** {result['lu3_filtered']}", ""]

report_path = report_dir / f"compare-{stamp}.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {report_path}")
