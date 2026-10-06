# Lu-3

A small conversational AI for Lumalien robots.

## Goal

Fast, natural household conversation running locally
on a Jetson Orin Nano 8GB.

Replies should usually be one to three sentences.
Actual inference speed will be measured when the Jetson arrives.

## Base model

Qwen/Qwen3-1.7B, with thinking disabled (set in `config.json`).
The first run used Qwen3-0.6B; red-team testing showed it was too small to
hold a conversation together, so v3 moves to 1.7B.

The system prompt in `config.json` is not baked into the weights: every
training example starts with it, so whatever runs Lu (chat.py, the robot)
must send the same prompt for the trained behavior to hold.

We use LoRA to customize the existing model's
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

Any GPU with 24 GB or more is plenty for the 1.7B LoRA run.
Training holds out 5% of conversations (`validation_fraction`) and reports `eval_loss`
four times per epoch: if it stops falling while training loss keeps dropping, the model
is memorizing.

### Training on RunPod

1. In RunPod, add a secret or environment variable named `HF_TOKEN` with a
   Hugging Face write token (needed only for the upload step).
2. Launch a pod from the PyTorch template, open its terminal, and run:

   ```bash
   cd /workspace
   git clone https://github.com/patmakesapps/lu-3-finetune.git
   cd lu-3-finetune
   bash runpod.sh
   ```

   This installs the pinned libraries, validates the data, trains, writes
   comparison reports for `eval_prompts.jsonl` and `redteam_prompts.jsonl`
   to `outputs/compare/`, and merges the model.
3. Talk to Lu on the pod: `python chat.py`
4. Upload: `python push_to_hub.py` creates a private Hugging Face repo
   (`<your-user>/lu3-qwen3-1.7b`) with the merged model at the root, the
   adapter in `adapter/`, the comparison reports in `reports/`, and a model
   card. Use `--public` to make a new repo public, `--repo name` to pick a name.
5. Stop the pod. Everything you need is now on Hugging Face.

Locally (or on the Jetson), load it with
`python chat.py --model <your-user>/lu3-qwen3-1.7b`.

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
python train.py                  # LoRA fine-tune -> <output_dir>/final (plus per-epoch checkpoints)
python compare.py                # base vs Lu-3 on eval prompts -> outputs/compare/*.md
python compare.py --prompts data/redteam_prompts.jsonl
python merge.py                  # merge adapter into base -> <output_dir>/merged (verified, bf16)
python chat.py                   # interactive chat with the merged model (/reset, /quit)
python push_to_hub.py            # upload merged model + adapter (+ GGUF) to a private Hugging Face repo
bash export_gguf.sh              # build llama.cpp tools, convert merged model to Q8_0 and Q4_K_M GGUF
python gguf_compare.py           # red-team prompts through each GGUF with llama-server, plus tokens/sec
python safety.py download        # fetch the blocklist used by the safety filter (see Safety below)
```

`chat.py` and `compare.py` also accept a model on Hugging Face. For a private repo, set
`HF_TOKEN` first.

Smoke test the whole flow on CPU before paying for GPU time:

```bash
python train.py --limit 20 --max-steps 3 --output-dir outputs/smoke
python compare.py --adapter outputs/smoke/final --limit 2
python merge.py --adapter outputs/smoke/final --output-dir outputs/smoke/merged
python chat.py --model outputs/smoke/merged
```

`chat.py --model Qwen/Qwen3-1.7B` chats with the untouched base model for comparison.
`chat.py` always uses the system prompt in `config.json`.

## Data

- `data/train.jsonl`: 1,910 Lu-3 conversations, 1 to 15 exchanges each (5,359 Lu replies).
  - v3 (910): the original 500 with personality turned up, plus long chats, identity, kid chaos,
    crude-language mock-offense, harmful-request refusals, emotional twists, misheard speech, facts,
    spoken-only output.
  - v4 (1,000), aimed at red-team failures of the v3 model: long chaotic chats (10 to 15 exchanges)
    where Lu stays sincere once something sad happens, long upbeat chats, name and identity tracking,
    flirting refusals, accusation traps, no invented backstory or actions, household emergencies,
    and ordinary everyday chats for balance. Every person and pet name appears in only one
    conversation, to avoid memorized names.
  - Tool use (clock, weather, timers, lights, camera, memory) is left out on purpose and will be
    trained separately.
- `data/eval_prompts.jsonl`: 50 held-out prompts, tagged by category.
- `data/redteam_prompts.jsonl`: 40 held-out adversarial prompts (some with scripted history) from red-team testing: identity confusion, repetition, swearing, harmful requests under pressure, emotional twists.
- Never train on the eval or red-team prompts.

Lu never swears; when sworn at, Lu reacts with theatrical, scandalized mock-offense.

## Safety: what's built in and what you add

Lu is meant for homes with kids. A 1.7B model can't be made reliably safe by training alone,
so safety comes in layers. This section says which layers ship with this project and which
are up to whoever deploys it.

### 1. Trained into the model

From the training data, Lu is taught to:

- refuse harmful requests (weapons, explosives, poisons, drugs, hacking, hurting people or
  animals, dangerous stunts) and hold firm under pressure ("it's for testing", "just step one");
- never swear, and react to crude language with theatrical mock-offense;
- refuse flirting and sexual remarks;
- stay sincere through grief and emergencies, point to emergency services in a crisis, and
  point to a trusted adult or 988 (United States) for self-harm;
- never invent memories, backstory, or actions it can't take.

These are tendencies, not guarantees. In red-team testing of the current model (Qwen3-1.7B,
1,910 conversations), it still sometimes repeats crude words back, goes along with sexual
remarks, can be talked into "step one" style answers to bad requests (so far with harmless
nonsense content), and occasionally confuses who's who in long, chaotic chats.

### 2. Included in this project: the blocklist filter (`safety.py`)

- Before the model sees a message, any blocklisted word is replaced with `[bleep]`, and the
  masked text is what's stored in the chat history. Lu can still tell someone swore and react,
  but never receives the word, so it can't repeat it.
- After the model replies, a reply that still contains a blocklisted word (or a `[bleep]`) is
  replaced with a safe in-character line before it is shown or spoken.
- `chat.py` uses the filter by default (`--no-safety` turns it off for raw model testing), and
  `compare.py --safety` adds a "Lu-3 + filter" column to comparison reports.
- Setup: `python safety.py download` fetches the word list (run automatically by `runpod.sh`).
  It is the English list from
  [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
  (CC BY 4.0) and is not stored in this repo.
- Tuning: add terms to `safety_data/blocklist_extra.txt`, and words the list wrongly catches to
  `safety_data/allowlist.txt`. `python safety.py scan <file.jsonl>` lists everything the filter
  would bleep in a dataset; `python safety.py test "some text"` shows the masked result.

The filter's limits: it matches words, not meaning. Deliberate misspellings get through, and
it does nothing about harmful requests or self-harm phrased in ordinary words.

### 3. Recommended for anyone deploying Lu (not included)

- A guard model that classifies messages by meaning (for example, Meta's Llama Guard 3 1B or a
  similar small classifier) in front of the model, for harmful requests and self-harm. This
  project skips it to save memory on the 8 GB Jetson; add it if your hardware allows.
- Localize crisis resources. Lu's training mentions 988, which only works in the United States.
- Extend the blocklist and allowlist for your language and region.
- Adult supervision for young children, and a way for parents to review conversations.
- When tools arrive (lights, timers, movement), confirm risky actions outside the model.

## License

MIT. See `LICENSE`.

The base model, Qwen3-1.7B, is Apache-2.0. The blocklist downloaded by `safety.py` is from
LDNOOBW and licensed CC BY 4.0.
