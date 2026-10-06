# Lu-3

A small conversational AI for Lumalien robots.

## Goal

Fast, natural household conversation running locally
on a Jetson Orin Nano 8GB.

Replies should usually be one to three sentences.
Actual inference speed will be measured when the Jetson arrives.

## Starting model

Qwen/Qwen3-0.6B, with thinking disabled.

We will use LoRA to customize the existing model's
conversational behavior.

## Pipeline

1. Prepare and validate conversation examples locally.
2. Convert conversations into training tokens.
3. Fine-tune with LoRA on RunPod.
4. Compare the original model and fine-tuned model.
5. Merge the LoRA adjustments into the model.
6. Export to GGUF and quantize.
7. Run and benchmark locally on the Jetson.

## RunPod training environment

Selected template: Runpod Pytorch 2.8.0

Container image:

`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

We will use the template's installed GPU-enabled PyTorch.

Additional training libraries are listed in requirements.txt:

- transformers==4.57.1
- peft==0.17.1
- accelerate==1.10.1

Before training, verify the installed Python and PyTorch
versions, CUDA availability, and successful Qwen model loading.
The selected environment has not been runtime-tested yet.

Prepare the scripts and data before launching paid GPU time.

## Local development

Project folder: C:\lu3 finetune

Local Python environment: .venv

Validate the conversation data from PowerShell:

```powershell
.\.venv\Scripts\python.exe validate_data.py
```

## Scripts

All settings come from `config.json`. Outputs go to `outputs/` (gitignored).

```bash
python train.py                  # LoRA fine-tune -> outputs/lu3/final (plus per-epoch checkpoints)
python compare.py                # base vs Lu-3 on eval prompts -> outputs/compare/*.md
python merge.py                  # merge adapter into base -> outputs/lu3/merged (verified, bf16)
python chat.py                   # interactive chat with the merged model (/reset, /quit)
```

Smoke test the whole flow on CPU before paying for GPU time:

```bash
python train.py --limit 20 --max-steps 3 --output-dir outputs/smoke
python compare.py --adapter outputs/smoke/final --limit 2
python merge.py --adapter outputs/smoke/final --output-dir outputs/smoke/merged
python chat.py --model outputs/smoke/merged
```

`chat.py --model Qwen/Qwen3-0.6B` chats with the untouched base model for comparison.

## Data

- `data/train.jsonl`: 500 Lu-3 conversations (350 single-turn, 150 multi-turn) for personality and conversation style. Tool use (clock, weather, timers, lights, camera, memory) is left out on purpose and will be trained separately.
- `data/eval_prompts.jsonl`: 50 held-out prompts, tagged by category, for comparing the base and fine-tuned models. Never train on these.

## License

MIT. See `LICENSE`.
