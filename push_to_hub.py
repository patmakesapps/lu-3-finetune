import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

output_dir = project_dir / config["output_dir"]
default_name = "lu3-" + config["base_model"].split("/")[-1].lower()

parser = argparse.ArgumentParser(description="Upload the merged Lu-3 model and adapter to a Hugging Face repo.")
parser.add_argument("--repo", default=default_name, help="Repo name, or user/name. Created if it doesn't exist.")
parser.add_argument("--public", action="store_true", help="Make a new repo public (default: private).")
parser.add_argument("--model-dir", default=str(output_dir / "merged"))
parser.add_argument("--adapter-dir", default=str(output_dir / "final"))
args = parser.parse_args()

token_file = project_dir / "hugging_face_token.txt"
token = os.environ.get("HF_TOKEN", "").strip()
if not token and token_file.exists():
    # The file may hold notes alongside the token; use the first hf_ word.
    token = next((word for word in token_file.read_text(encoding="utf-8").split() if word.startswith("hf_")), "")
if not token:
    raise SystemExit("Set HF_TOKEN (a Hugging Face write token) or create hugging_face_token.txt.")

model_dir, adapter_dir = Path(args.model_dir), Path(args.adapter_dir)
if not (model_dir / "model.safetensors").exists() and not list(model_dir.glob("model-*.safetensors")):
    raise SystemExit(f"No merged model in {model_dir}. Run merge.py first.")

api = HfApi(token=token)
username = api.whoami()["name"]
repo_id = args.repo if "/" in args.repo else f"{username}/{args.repo}"
api.create_repo(repo_id, private=not args.public, exist_ok=True)

with (project_dir / "data" / "train.jsonl").open(encoding="utf-8") as file:
    conversation_count = sum(1 for line in file if line.strip())

model_card = f"""---
base_model: {config["base_model"]}
library_name: transformers
license: apache-2.0
tags:
- lora
- conversational
- robotics
---

# Lu-3 ({config["base_model"]} fine-tune)

Lu-3, usually called Lu, is a household robot companion from Lumalien: theatrical, cheeky, loyal,
and short-spoken, built for voice conversation. This is {config["base_model"]} with a LoRA adapter
merged in, trained on {conversation_count} original conversations. The unmerged adapter is in `adapter/`,
and GGUF files for llama.cpp (Q8_0, Q4_K_M) are in `gguf/` when exported.

Tool use (time, weather, timers, lights, camera, memory) is not trained yet.

## Safety

Lu is trained to refuse harmful requests, never swear, and turn down flirting, but a small model
does not do this reliably under adversarial use: it can repeat crude words back and sometimes
plays along. No content filter ships with it. See the source repo's README for known limitations
and what deployers should add, such as an input and output filter or a guard model.

Age guard: whenever a user says they are under 18 (`age_guard.py` in the source repo detects
this), clear the conversation history and append this child note to the system prompt for the
rest of the session. Kid conversations were trained with exactly this text:

```
{config["child_note"]}
```

## Running the GGUF with llama.cpp

`gguf/lu3-q4_k_m.gguf` is the intended size for a Jetson Orin Nano 8GB (`gguf/lu3-q8_0.gguf` is
higher quality but larger). Render the prompt with this repo's tokenizer chat template
(system prompt, history, `add_generation_prompt=True`, `enable_thinking=False`) and send the
text to llama-server's `/completion` endpoint with `"stop": ["<|im_end|>"]` and the sampling
settings below; `gguf_compare.py` in the source repo does exactly this.

## Use

Always pass this system prompt and disable thinking:

```
{config["system_prompt"]}
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
messages = [{{"role": "system", "content": SYSTEM_PROMPT}}, {{"role": "user", "content": "hey lu"}}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

Sampling used in testing: temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty {config["repetition_penalty"]}, max_new_tokens {config["max_new_tokens"]}.

## Training

LoRA rank {config["lora_rank"]}, alpha {config["lora_alpha"]}, learning rate {config["learning_rate"]},
{config["epochs"]} epochs, loss on assistant replies only.
Source: https://github.com/patmakesapps/lu-3-finetune
"""

print(f"Uploading merged model from {model_dir} to {repo_id} ...")
api.upload_folder(repo_id=repo_id, folder_path=str(model_dir), commit_message="Upload merged Lu-3 model")

if adapter_dir.exists():
    print(f"Uploading adapter from {adapter_dir} ...")
    api.upload_folder(repo_id=repo_id, folder_path=str(adapter_dir), path_in_repo="adapter",
                      commit_message="Upload LoRA adapter")

gguf_dir = output_dir / "gguf"
if list(gguf_dir.glob("*.gguf")):
    print(f"Uploading GGUF files from {gguf_dir} ...")
    api.upload_folder(repo_id=repo_id, folder_path=str(gguf_dir), path_in_repo="gguf",
                      allow_patterns=["*.gguf"], ignore_patterns=["*bf16*"],
                      commit_message="Upload GGUF files for llama.cpp")

compare_dir = project_dir / "outputs" / "compare"
if compare_dir.exists():
    api.upload_folder(repo_id=repo_id, folder_path=str(compare_dir), path_in_repo="reports",
                      allow_patterns=["*.md"], commit_message="Upload comparison reports")

api.upload_file(repo_id=repo_id, path_or_fileobj=model_card.encode("utf-8"), path_in_repo="README.md",
                commit_message="Add model card")
print(f"Done: https://huggingface.co/{repo_id}")
