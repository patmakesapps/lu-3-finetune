#!/usr/bin/env bash
# Run on a RunPod PyTorch pod from the repo root:
#   git clone https://github.com/patmakesapps/lu-3-finetune.git && cd lu-3-finetune && bash runpod.sh
# Then test with `python chat.py` and upload with `python push_to_hub.py`.
set -euo pipefail

python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pip install -q -r requirements.txt hf_transfer  # RunPod enables HF_HUB_ENABLE_HF_TRANSFER

mkdir -p outputs
python validate_data.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py 2>&1 | tee outputs/train.log
python compare.py
python compare.py --prompts data/redteam_prompts.jsonl
python merge.py

echo
echo "Done. Next:"
echo "  python chat.py            # talk to Lu on the pod"
echo "  python push_to_hub.py     # upload to a private Hugging Face repo (needs HF_TOKEN)"
