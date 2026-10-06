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