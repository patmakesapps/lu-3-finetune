"""Chat with a GGUF build of Lu through llama.cpp's llama-server, the way the Jetson will run it.

Same behavior as chat.py (system prompt, history, age guard, /reset), but generation goes
through llama-server instead of transformers, so it exercises the quantized file itself.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from age_guard import stated_minor

project_dir = Path(__file__).resolve().parent

with (project_dir / "config.json").open(encoding="utf-8") as file:
    config = json.load(file)

output_dir = project_dir / config["output_dir"]

parser = argparse.ArgumentParser(description="Chat with a GGUF build of Lu through llama-server.")
parser.add_argument("--gguf", default=str(output_dir / "gguf" / "lu3-q4_k_m.gguf"))
parser.add_argument("--server", default=str(project_dir / "llama.cpp" / "build" / "bin" / "llama-server"))
parser.add_argument("--tokenizer", default=str(output_dir / "merged"),
                    help="Folder or Hugging Face id with the chat template (only used to render prompts).")
parser.add_argument("--port", type=int, default=8091)
parser.add_argument("--history", type=int, default=config["chat_history_exchanges"])
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

print(f"Starting llama-server with {Path(args.gguf).name} ...")
server = subprocess.Popen(
    [args.server, "-m", args.gguf, "--port", str(args.port), "-c", "4096", "--no-webui"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def post(payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


try:
    for _ in range(300):
        if server.poll() is not None:
            raise SystemExit(f"llama-server exited early with code {server.returncode}.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit("llama-server did not become ready.")

    print(f"Chatting with {Path(args.gguf).name}. Commands: /reset, /quit")
    history, child_mode = [], False

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            break
        if user_text == "/reset":
            history, child_mode = [], False
            print("(history cleared)")
            continue

        # Same hard stop as chat.py: forget the conversation and keep the child note on.
        if not child_mode and stated_minor(user_text):
            history, child_mode = [], True
            print("(under 18 stated: history cleared, child mode on)")

        system_prompt = config["system_prompt"] + (" " + config["child_note"] if child_mode else "")
        history.append({"role": "user", "content": user_text})
        history = history[-(args.history * 2 - 1):]
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        result = post({
            "prompt": prompt,
            "n_predict": config["max_new_tokens"],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": config["repetition_penalty"],
            "stop": ["<|im_end|>"],
        })
        reply = result["content"].strip()
        history.append({"role": "assistant", "content": reply})
        timings = result.get("timings", {})
        print(f"Lu: {reply}")
        print(f"({timings.get('predicted_n', 0)} tokens, {timings.get('predicted_per_second', 0):.1f} tok/s)")
finally:
    server.terminate()
    server.wait()
