import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

settings = config["dpo"]

parser = argparse.ArgumentParser(description="DPO preference training on top of the fine-tuned Lu-3 model.")
parser.add_argument("--model", default=settings["sft_model"], help="Fine-tuned (merged) model: local folder or Hugging Face id.")
parser.add_argument("--data", default=str(project_dir / settings["data"]))
parser.add_argument("--output-dir", default=str(project_dir / settings["output_dir"]))
parser.add_argument("--limit", type=int, default=None, help="Use only the first N pairs (smoke tests).")
parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke tests).")
args = parser.parse_args()

random.seed(config["seed"])
tokenizer = AutoTokenizer.from_pretrained(args.model)


def render(pair):
    # Same prompt text as training and inference: system prompt, history, and the
    # generation prompt with thinking disabled. TRL appends <|im_end|> to each reply.
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": config["system_prompt"]}] + pair["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt": prompt, "chosen": pair["chosen"], "rejected": pair["rejected"]}


with Path(args.data).open(encoding="utf-8") as file:
    pairs = [json.loads(line) for line in file if line.strip()]
if args.limit:
    pairs = pairs[: args.limit]

random.shuffle(pairs)
held_out = int(len(pairs) * settings["validation_fraction"])
validation = Dataset.from_list([render(p) for p in pairs[:held_out]]) if held_out else None
training = Dataset.from_list([render(p) for p in pairs[held_out:]])

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if use_bf16 else torch.float32)

steps_per_epoch = max(1, len(training) // (settings["batch_size"] * settings["gradient_accumulation_steps"]))
print(f"Pairs: {len(training)} training, {held_out} validation | model: {args.model}")

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # with a LoRA adapter, the reference is the same model with the adapter turned off
    args=DPOConfig(
        output_dir=args.output_dir,
        beta=settings["beta"],
        num_train_epochs=settings["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps" if validation else "no",
        eval_steps=max(1, steps_per_epoch // 2),
        save_strategy="epoch",
        max_prompt_length=config["max_seq_length"] - 128,
        max_length=config["max_seq_length"],
        bf16=use_bf16,
        seed=config["seed"],
        report_to="none",
    ),
    train_dataset=training,
    eval_dataset=validation,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)
trainer.train()

final_dir = Path(args.output_dir) / "final"
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Saved DPO adapter to {final_dir}")
