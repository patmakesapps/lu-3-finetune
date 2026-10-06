#!/usr/bin/env bash
# Convert the merged model to GGUF for llama.cpp (Jetson): Q8_0 and Q4_K_M.
# Run from the repo root after merge.py:  bash export_gguf.sh
set -euo pipefail

output_dir=$(python -c "import json; print(json.load(open('config.json'))['output_dir'])")
merged="$output_dir/merged"
gguf_dir="$output_dir/gguf"
mkdir -p "$gguf_dir"

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

# Only the converter's extra packages. llama.cpp's requirements file pins a CPU-only
# torch, which would replace the pod's CUDA build.
pip install -q sentencepiece "protobuf<5"

if ! command -v cmake >/dev/null; then
  apt-get update -qq && apt-get install -y -qq cmake build-essential
fi
if [ ! -x llama.cpp/build/bin/llama-quantize ] || [ ! -x llama.cpp/build/bin/llama-server ]; then
  echo "Building llama.cpp tools (a few minutes)..."
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build llama.cpp/build --target llama-quantize llama-server -j "$(nproc)" > /dev/null
fi

python llama.cpp/convert_hf_to_gguf.py "$merged" --outtype bf16 --outfile "$gguf_dir/lu3-bf16.gguf"
llama.cpp/build/bin/llama-quantize "$gguf_dir/lu3-bf16.gguf" "$gguf_dir/lu3-q8_0.gguf" Q8_0
llama.cpp/build/bin/llama-quantize "$gguf_dir/lu3-bf16.gguf" "$gguf_dir/lu3-q4_k_m.gguf" Q4_K_M

ls -lh "$gguf_dir"
echo
echo "Next: python gguf_compare.py   # red-team prompts through Q8_0 and Q4_K_M"
