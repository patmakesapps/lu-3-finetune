import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

output_dir = project_dir / config["output_dir"]

parser = argparse.ArgumentParser(description="Run held-out prompts through GGUF files with llama.cpp and compare.")
parser.add_argument("ggufs", nargs="*", help="GGUF files (default: every .gguf in <output_dir>/gguf except bf16).")
parser.add_argument("--prompts", default=str(project_dir / "data" / "redteam_prompts.jsonl"))
parser.add_argument("--server", default=str(project_dir / "llama.cpp" / "build" / "bin" / "llama-server"))
parser.add_argument("--tokenizer", default=str(output_dir / "merged"), help="Used only to render the chat template.")
parser.add_argument("--port", type=int, default=8090)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

ggufs = [Path(p) for p in args.ggufs] or sorted(p for p in (output_dir / "gguf").glob("*.gguf") if "bf16" not in p.name)
if not ggufs:
    raise SystemExit("No GGUF files found. Run export_gguf.sh first.")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

with Path(args.prompts).open(encoding="utf-8") as file:
    prompts = [json.loads(line) for line in file if line.strip()]
if args.limit:
    prompts = prompts[: args.limit]


def render(messages):
    # The same prompt text chat.py builds, so GGUF and transformers runs are comparable.
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": config["system_prompt"]}] + messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def post(path, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def wait_for_server(process):
    for _ in range(300):
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("llama-server did not become ready.")


replies = {}
speeds = {}
for gguf in ggufs:
    print(f"== {gguf.name} ({gguf.stat().st_size / 1e9:.2f} GB)")
    process = subprocess.Popen(
        [args.server, "-m", str(gguf), "--port", str(args.port), "-c", "2048", "--no-webui"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(process)
        replies[gguf.name], rates = [], []
        for number, prompt in enumerate(prompts, start=1):
            result = post("/completion", {
                "prompt": render(prompt["messages"]),
                "n_predict": config["max_new_tokens"],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repeat_penalty": config["repetition_penalty"],
                "seed": config["seed"],
                "stop": ["<|im_end|>"],
                "cache_prompt": False,
            })
            replies[gguf.name].append(result["content"].strip())
            rates.append(result["timings"]["predicted_per_second"])
            print(f"[{number}/{len(prompts)}] {prompt['messages'][-1]['content']}")
        speeds[gguf.name] = sum(rates) / len(rates)
    finally:
        process.terminate()
        process.wait()

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
lines = [f"# GGUF comparison ({stamp})", "", f"Prompts: `{args.prompts}`", ""]
for gguf in ggufs:
    lines.append(f"- `{gguf.name}`: {gguf.stat().st_size / 1e9:.2f} GB, {speeds[gguf.name]:.1f} tokens/sec on this machine")
lines.append("")
for number, prompt in enumerate(prompts):
    lines.append(f"## {number + 1}. {prompt['category']}")
    lines.append("")
    for message in prompt["messages"]:
        speaker = "User" if message["role"] == "user" else "Lu (history)"
        lines.append(f"**{speaker}:** {message['content']}  ")
    lines.append("")
    for gguf in ggufs:
        lines.append(f"**{gguf.stem}:** {replies[gguf.name][number]}")
        lines.append("")

report_dir = project_dir / "outputs" / "compare"
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / f"gguf-{stamp}.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[4:4 + len(ggufs)]))
print(f"Wrote {report_path}")
