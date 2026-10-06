#!/usr/bin/env bash
# DPO round on a RunPod PyTorch pod, from the repo root:
#   export HF_TOKEN=hf_...      (the fine-tuned model on Hugging Face is private)
#   bash runpod_dpo.sh
# Then test with the printed chat command and upload with push_to_hub.py.
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Set HF_TOKEN first: export HF_TOKEN=hf_..."
  exit 1
fi

python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pip install -q -r requirements.txt hf_transfer  # RunPod enables HF_HUB_ENABLE_HF_TRANSFER

mkdir -p outputs
python safety.py download

dpo_dir=$(python -c "import json; print(json.load(open('config.json'))['dpo']['output_dir'])")
python dpo.py 2>&1 | tee outputs/dpo.log
# "Base" in this report is the fine-tuned model before DPO; "Lu-3" is after.
python compare.py --adapter "$dpo_dir/final" --prompts data/redteam_prompts.jsonl --safety
python compare.py --adapter "$dpo_dir/final"
python merge.py --adapter "$dpo_dir/final" --output-dir "$dpo_dir/merged"

echo
echo "Done. Next:"
echo "  python chat.py --model $dpo_dir/merged"
echo "  python push_to_hub.py --model-dir $dpo_dir/merged --adapter-dir $dpo_dir/final --repo lu3-qwen3-1.7b-dpo"
