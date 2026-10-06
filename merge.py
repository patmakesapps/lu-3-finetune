import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

parser = argparse.ArgumentParser(description="Merge the Lu-3 LoRA adapter into the base model.")
parser.add_argument("--adapter", default=str(project_dir / config["output_dir"] / "final"))
parser.add_argument("--output-dir", default=str(project_dir / config["output_dir"] / "merged"))
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.adapter)

# Merge in float32 for accuracy, then save in bfloat16 like the base model.
base = AutoModelForCausalLM.from_pretrained(config["base_model"], dtype=torch.float32)
model = PeftModel.from_pretrained(base, args.adapter).eval()

check_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": "Hi Lu, what are you up to?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
check_inputs = tokenizer(check_text, return_tensors="pt")

with torch.no_grad():
    adapter_logits = model(**check_inputs).logits

merged = model.merge_and_unload()

with torch.no_grad():
    merged_logits = merged(**check_inputs).logits

difference = (adapter_logits - merged_logits).abs().max().item()
print(f"Max logit difference, adapter vs merged: {difference:.6f}")
if difference > 1e-3:
    raise ValueError("Merged model does not match the adapter model.")

merged = merged.to(torch.bfloat16)
merged.save_pretrained(args.output_dir, safe_serialization=True)
tokenizer.save_pretrained(args.output_dir)
print(f"Saved merged model to {args.output_dir}")

reloaded = AutoModelForCausalLM.from_pretrained(args.output_dir, dtype=torch.float32).eval()
with torch.no_grad():
    output = reloaded.generate(
        **check_inputs,
        max_new_tokens=config["max_new_tokens"],
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
reply = tokenizer.decode(output[0][check_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print(f"Reloaded merged model says: {reply}")
