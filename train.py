import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

project_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3 into Lu-3.")
parser.add_argument("--data", default=str(project_dir / "data" / "train.jsonl"))
parser.add_argument("--output-dir", default=None, help="Defaults to output_dir in config.json.")
parser.add_argument("--limit", type=int, default=None, help="Use only the first N conversations (smoke tests).")
parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (smoke tests).")
args = parser.parse_args()

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

output_dir = Path(args.output_dir or project_dir / config["output_dir"])
random.seed(config["seed"])
torch.manual_seed(config["seed"])

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
system_message = {"role": "system", "content": config["system_prompt"]}


def build_examples(conversation):
    """One training example per assistant turn.

    The prompt is rendered exactly as at inference time (history, then the
    generation prompt with thinking disabled), and only the reply is scored.
    Rendering the whole conversation at once would not match inference,
    because Qwen3's template drops the empty think block from earlier turns.
    """
    messages = [system_message] + conversation["messages"]
    examples = []

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        prompt_text = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        reply_ids = tokenizer(message["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + reply_ids
        if len(input_ids) > config["max_seq_length"]:
            raise ValueError(f"Example is {len(input_ids)} tokens, over max_seq_length.")

        examples.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + reply_ids,
        })

    return examples


with Path(args.data).open(encoding="utf-8") as file:
    conversations = [json.loads(line) for line in file if line.strip()]

if args.limit:
    conversations = conversations[: args.limit]

# Hold out whole conversations so validation loss shows memorization:
# it stops falling (or rises) while training loss keeps dropping.
held_out = int(len(conversations) * config["validation_fraction"])
shuffled = random.sample(conversations, len(conversations))
validation_conversations, training_conversations = shuffled[:held_out], shuffled[held_out:]

examples = [example for conversation in training_conversations for example in build_examples(conversation)]
validation_examples = [example for conversation in validation_conversations for example in build_examples(conversation)]
random.shuffle(examples)

# The prompt for a final turn must match what the chat template produces for
# the full conversation, or training and inference formats have drifted apart.
sample = [system_message] + conversations[0]["messages"]
expected = tokenizer.apply_chat_template(sample, tokenize=False, enable_thinking=False)
rebuilt = tokenizer.apply_chat_template(
    sample[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
) + sample[-1]["content"] + "<|im_end|>\n"
if expected != rebuilt:
    raise ValueError("Training format does not match the Qwen3 chat template.")


def collate(batch):
    length = max(len(example["input_ids"]) for example in batch)
    pad_id = tokenizer.pad_token_id

    input_ids, labels, attention_mask = [], [], []
    for example in batch:
        padding = length - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_id] * padding)
        labels.append(example["labels"] + [-100] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(
    config["base_model"], dtype=torch.bfloat16 if use_bf16 else torch.float32
)
model = get_peft_model(model, LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
))

reply_tokens = sum(sum(label != -100 for label in example["labels"]) for example in examples)
print(f"Conversations: {len(training_conversations)} training, {len(validation_conversations)} validation")
print(f"Training examples (assistant turns): {len(examples)}, validation examples: {len(validation_examples)}")
print(f"Longest example: {max(len(example['input_ids']) for example in examples)} tokens")
print(f"Scored reply tokens: {reply_tokens}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}, bf16: {use_bf16}")
model.print_trainable_parameters()

steps_per_epoch = math.ceil(len(examples) / (config["batch_size"] * config["gradient_accumulation_steps"]))

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps" if validation_examples else "no",
        eval_steps=max(1, steps_per_epoch // 4),
        per_device_eval_batch_size=config["batch_size"],
        save_strategy="epoch",
        bf16=use_bf16,
        seed=config["seed"],
        report_to="none",
        remove_unused_columns=False,
    ),
    train_dataset=examples,
    eval_dataset=validation_examples or None,
    data_collator=collate,
)
trainer.train()

final_dir = output_dir / "final"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Saved LoRA adapter to {final_dir}")
