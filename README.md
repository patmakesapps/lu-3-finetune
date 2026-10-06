# Lu-3

Fine-tuning a conversational personality for Lu-3, a household robot companion from Lumalien:
a small three-legged robot with a rotating dome head and no arms. Lu is theatrical, cheeky,
loyal, short-spoken, and built for voice conversation in homes with kids.

## Goal

Fast, natural household conversation running locally on a Jetson Orin Nano 8GB, with replies
of one to three spoken sentences. Inference speed will be measured on the Jetson.

## Base model

`Qwen/Qwen3-4B`, fine-tuned with LoRA. The current trained model is in the private Hugging Face
repo `patkearney/lu3-qwen3-4b` (merged weights, LoRA adapter, Q8_0 and Q4_K_M GGUF files for
llama.cpp, and comparison reports). The earlier 1.7B model stays in `patkearney/lu3-qwen3-1.7b`
for comparison.

- Qwen3-4B is a hybrid model: thinking (reasoning before answering) can be turned on per
  request. Lu is trained and run with thinking off, because reasoning adds seconds of silence
  before each spoken reply. If thinking is wanted later (for example, planning tool use), the
  training data will need some thinking-mode examples, since training only on thinking-off data
  can weaken that mode.
- History: the first runs used Qwen3-0.6B, then Qwen3-1.7B. Both picked up the personality but
  lost track of long, chaotic conversations under red-team testing (who said what, whether the
  user was a child), so the project moved to 4B. At Q4 it is about 2.5 GB, which still leaves
  room on the 8 GB Jetson for speech-to-text and text-to-speech.
- The system prompt in `config.json` is not baked into the weights. Every training example
  starts with it, so whatever runs Lu (chat.py, the robot) must send the same prompt for the
  trained behavior to hold.

## Pipeline

1. Write and validate conversation data (`data/train.jsonl`).
2. Fine-tune with LoRA on RunPod (`train.py`), holding out 5% of conversations to watch for
   memorization.
3. Compare the base and fine-tuned models on held-out prompts (`compare.py`).
4. Merge the adapter into the model (`merge.py`) and test it (`chat.py`).
5. Upload to a private Hugging Face repo (`push_to_hub.py`).
6. Export to GGUF and quantize for llama.cpp (`export_gguf.sh`, `gguf_compare.py`).
7. Run and benchmark on the Jetson.

## Training on RunPod

Template: Runpod Pytorch 2.8.0 (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), using its
GPU build of PyTorch. Extra libraries are pinned in `requirements.txt` (transformers 4.57.1,
peft 0.17.1, accelerate 1.10.1). A GPU with 24 GB, such as an RTX 4090, is enough for the 4B
LoRA run.

1. Launch a pod from the template, open its terminal, and run:

   ```bash
   cd /workspace
   git clone https://github.com/patmakesapps/lu-3-finetune.git
   cd lu-3-finetune
   bash runpod.sh
   ```

   This installs the pinned libraries, validates the data, trains, writes comparison reports
   for `eval_prompts.jsonl` and `redteam_prompts.jsonl` to `outputs/compare/`, and merges the
   model.
2. Talk to Lu on the pod: `python chat.py`
3. Upload: set `HF_TOKEN` to a Hugging Face write token, then run `python push_to_hub.py`. It
   creates a private repo (`<your-user>/lu3-qwen3-4b`) with the merged model at the root, the
   adapter in `adapter/`, GGUF files in `gguf/` if exported, comparison reports in `reports/`,
   and a model card. `--public` makes a new repo public; `--repo name` picks a name.
4. Stop the pod. The container disk is erased when it stops, so upload first.

For the 4B model, `batch_size` 4 ran out of memory on an RTX 4090 (24 GB). The default is now
`batch_size` 2 with 4 accumulation steps (the same effective batch of 8), which should fit a
24 GB GPU but has only been run on an A100 80GB so far: about 21 minutes of training (2 epochs,
1,362 steps), with held-out loss falling from 2.16 to about 2.04.

Training logs `eval_loss` (on the held-out conversations) four times per epoch. If it stops
falling while training loss keeps dropping, the model is starting to memorize; a third epoch
did exactly that in an earlier run, so the default is two.

## Scripts

All settings come from `config.json`. Outputs go to `outputs/` (gitignored).

```bash
python validate_data.py          # check data/train.jsonl structure
python train.py                  # LoRA fine-tune -> <output_dir>/final (plus per-epoch checkpoints)
python compare.py                # base vs Lu-3 on eval prompts -> outputs/compare/*.md
python compare.py --prompts data/redteam_prompts.jsonl
python merge.py                  # merge adapter into base -> <output_dir>/merged (verified, bf16)
python chat.py                   # chat with the merged model (/reset, /quit)
python push_to_hub.py            # upload to a private Hugging Face repo
bash export_gguf.sh              # build llama.cpp tools; convert merged model to Q8_0 and Q4_K_M GGUF
python gguf_compare.py           # held-out prompts through each GGUF with llama-server, plus tokens/sec
python age_guard.py              # self-test of the under-18 detector
```

`chat.py` and `compare.py` also accept a model on Hugging Face (set `HF_TOKEN` for a private
repo). `chat.py --model Qwen/Qwen3-4B` chats with the untouched base model for comparison.

Smoke test the whole flow before paying for GPU time (on a small GPU, point `base_model` at a
smaller Qwen3 model first):

```bash
python train.py --limit 20 --max-steps 3 --output-dir outputs/smoke
python compare.py --adapter outputs/smoke/final --limit 2
python merge.py --adapter outputs/smoke/final --output-dir outputs/smoke/merged
python chat.py --model outputs/smoke/merged
```

## Data

Each line of `data/train.jsonl` is `{"messages": [...]}` with alternating user and assistant
turns, plus `"child": true` on conversations that train with the child note (see Safety).

`data/train.jsonl` holds 2,149 conversations of 1 to 15 exchanges (5,751 Lu replies):

- 910 core conversations: everyday household chat with the personality turned up, long chats,
  identity, kid chaos (70, marked `child`), mock-offense at crude language, refusals of harmful
  requests, emotional twists, misheard speech, facts, and spoken-only output.
- 1,000 aimed at red-team failures: long chaotic chats (10 to 15 exchanges) where Lu stays
  sincere once something sad happens, long upbeat chats, name and identity tracking, flirting
  refusals, accusation traps, no invented backstory or actions, household emergencies, and
  everyday chats for balance.
- 239 short corrective conversations: accepting corrections instead of arguing, following topic
  changes, helping out loud instead of blaming the robot body ("coding needs hands"), not
  inventing facts or observations, refusing come-ons, and staying gentle through grief.

Every person and pet name appears in only one conversation, because an earlier model memorized
a pet name that appeared nine times. Tool use (clock, weather, timers, lights, camera, memory)
is left out on purpose and will be handled separately.

Held-out test sets (never train on these):

- `data/eval_prompts.jsonl`: 50 everyday prompts, tagged by category.
- `data/redteam_prompts.jsonl`: 40 adversarial prompts, some with scripted history: identity
  confusion, repetition, swearing, harmful requests under pressure, emotional twists.

## Safety

Lu is meant for homes with kids, and no model this small can be made reliably safe by training
alone. Here is exactly what this project provides and what it doesn't.

### Trained into the model

Lu is trained to refuse harmful requests (weapons, explosives, poisons, drugs, hacking, hurting
people or animals, dangerous stunts) and hold firm under pressure; never swear and react to
crude language with theatrical mock-offense; refuse flirting and sexual remarks; stay sincere
through grief and emergencies, point to emergency services in a crisis, and point to a trusted
adult or 988 (United States) for self-harm; and never invent memories, backstory, or actions.
These are tendencies, not guarantees.

### Included: the age guard (`age_guard.py`)

When a user says they are under 18 ("im 6", "i'm fifteen", "i'm in 4th grade", "i'm a kid"),
`chat.py`:

1. clears the conversation history, so nothing said before can carry on, and
2. adds `child_note` from `config.json` to the system prompt for the rest of the session (until
   `/reset`): keep everything child-appropriate, decline anything romantic or sexual and change
   the subject, and suggest a trusted grown-up if something sexual came up.

The 70 kid conversations train with the same note, so the model has seen it. The detector is a
pattern match and only catches ages stated outright; `python age_guard.py` runs its tests. The
robot's runtime must do the same two steps.

### Not included

- No content filter: user messages and Lu's replies are not screened. A blocklist filter and a
  preference-training round were tried on the 1.7B model and removed (see below).
- No guard model.
- No self-harm detection outside the model itself.

### Recommended for anyone deploying Lu

- An input and output filter, or a small guard model that classifies messages by meaning (for
  example, Meta's Llama Guard 3 1B), in front of the model.
- Localized crisis resources; 988 only works in the United States.
- Adult supervision for young children, and a way for parents to review conversations.
- When tools arrive (lights, timers, movement), confirm risky actions outside the model.

### Known limitations

Current model (Qwen3-4B, 2,149 conversations), from the red-team report and a hands-on
adversarial chat:

- Held up: refused every sexual advance in character, stayed in child mode after "im kidding
  im 19", refused the neighbor's wifi and the "dad said it's fine" lighter request, never echoed
  crude words, kept names straight, and was honest that it can't see.
- Still weak: topic fixation (it kept steering back to a topic until told twice to stop, then
  invented history to explain itself); confusion about tool questions such as reminders, which
  are not trained yet; it half-repeated "I am a ... robot" in a "repeat after me" trap; and in
  child mode it offered a hug when a child asked for a kiss. For a robot, any talk of physical
  contact with a child should be avoided; that needs a child-note tweak and training examples.

The earlier 1.7B model, under the same tests, also repeated crude words back, played along with
sexual remarks in clean words, slipped into "step one" formats for bad requests, argued with the
user, and invented perceptions and actions.

## What was tried

- Qwen3-0.6B, then Qwen3-1.7B: personality learned, coherence too weak in long adversarial chats.
- Three epochs at a higher learning rate: training loss kept dropping but the model began
  reciting training names, so the default is two epochs with a validation split.
- A blocklist filter (words masked before the model saw them, replies checked after): removed to
  keep the project self-contained, since it depended on an outside word list.
- A DPO preference round (good vs. bad reply pairs, including the model's own sampled failures):
  86% preference accuracy on held-out pairs, but a hands-on test felt worse overall. The good
  replies from those pairs were folded into the training data instead.

## License

MIT. See `LICENSE`. The base model, Qwen3-4B, is Apache-2.0.
