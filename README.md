# Lu-3

Fine-tuning a conversational personality for Lu-3, a household robot companion from Lumalien:
a small three-legged robot with a rotating dome head and no arms. Lu is theatrical, cheeky,
loyal, short-spoken, and built for voice conversation in homes with kids.

## Goal

Fast, natural household conversation running locally on a Jetson Orin Nano 8GB, with replies
of one to three spoken sentences. Inference speed will be measured on the Jetson.

## Base model

`Qwen/Qwen3-4B`, fine-tuned with LoRA. The trained model is in the private Hugging Face repo
`patkearney/lu3-qwen3-4b` (merged weights, LoRA adapter, Q8_0 and Q4_K_M GGUF files for
llama.cpp, and comparison reports). It currently holds the tool-use round (`get_time`,
`get_machine_info`); the memory-tools round (see Data) is being trained and replaces it once
uploaded. The earlier 1.7B model stays in `patkearney/lu3-qwen3-1.7b` for comparison.

Q4_K_M (2.4 GB) is the intended Jetson build: in a hands-on chat through llama-server
(`chat_gguf.py`) its personality and refusals held up like the full model's. Jetson speed is
still to be measured.

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
- Every training example also carries the tools section that Qwen3's chat template adds to the
  system prompt, because the Lu-3 runtime sends its tools on every turn. The runtime's seven
  tools (`tools` in `config.json`) must stay identical in `data/tools.json` to
  `TOOL_DEFINITIONS` in the runtime (`software/lu3/tools.py` in the Lu-3 repo); the template
  writes them into the prompt word for word.
- The runtime also adds up to five saved memories matching the latest message to the end of the
  system prompt ("Things you remember: ..."). Training builds the same prompt per user turn from
  each message's `memories` field, so the two must stay in step: system prompt, then the child
  note if on, then the memories.

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

`RUNPOD_CHEATSHEET.txt` has the commands below in copy-paste form.

1. Launch a pod from the template, open its terminal, and run:

   ```bash
   cd /workspace
   export HF_HOME=/workspace/hf_cache
   git clone https://github.com/patmakesapps/lu-3-finetune.git
   cd lu-3-finetune
   bash runpod.sh
   ```

   `HF_HOME` keeps the base model download on the `/workspace` volume instead of the small
   container disk (a full run with GGUF export uses about 35 GB). `runpod.sh` installs the
   pinned libraries, validates the data, trains, writes comparison reports for
   `eval_prompts.jsonl` and `redteam_prompts.jsonl` to `outputs/compare/`, and merges the model.
2. Talk to Lu on the pod: `python chat.py` (it doesn't send tools; test tool use in the Lu-3
   runtime).
3. Export GGUF files for the Jetson: `bash export_gguf.sh`
4. Upload: set `HF_TOKEN` to a Hugging Face write token, then run `python push_to_hub.py`. It
   creates a private repo (`<your-user>/lu3-qwen3-4b`) with the merged model at the root, the
   adapter in `adapter/`, GGUF files in `gguf/` if exported, comparison reports in `reports/`,
   and a model card. `--public` makes a new repo public; `--repo name` picks a name.
5. Stop the pod. The container disk is erased when it stops, so upload first.

For the 4B model, `batch_size` 4 ran out of memory on an RTX 4090 (24 GB). The default is now
`batch_size` 2 with 4 accumulation steps (the same effective batch of 8), which should fit a
24 GB GPU but has only been run on an A100 80GB so far. The tool-use round trained in about 21
minutes (2 epochs, 1,362 steps), with held-out loss falling from 2.16 to about 2.04. The
memory-tools round takes about 65 minutes (2,528 steps at about 1.6 seconds each): more
examples, and every one carries the seven-tool section (about 720 tokens of prompt, against 365
with two tools).

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
python chat_gguf.py              # chat with the Q4_K_M GGUF through llama-server, like the Jetson (--gguf for Q8_0)
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

Each line of `data/train.jsonl` is `{"messages": [...]}`, plus `"child": true` on conversations
that train with the child note (see Safety) and an optional `"tools": [names]` (default:
`tools` in `config.json`, the runtime's seven). Messages run user, then a spoken reply or one or
more tool call and tool result pairs ending in a spoken reply, in the OpenAI format the runtime
uses: an assistant message with empty `content` and one `tool_calls` entry (arguments as a JSON
string), then a `tool` message with the result as a JSON string. Tool calls are scored as the
template writes them; tool results are never scored. A user message may carry `"memories": [...]`:
the runtime's automatic lookup for that message, added to the system prompt after "Things you
remember:" for every request made while answering it, exactly as the runtime's `brain.py` does.

`data/train.jsonl` holds 3,209 conversations (9,040 spoken Lu replies, 1,643 tool calls).

The memory tools changed the default tool list from two tools to seven. 1,352 of the older
conversations mention lasting personal facts (names, pets, jobs, birthdays) or memory without
saving anything, which would teach "don't save" now that `remember` is listed, so they are
pinned to the old two-tool list (`get_time`, `get_machine_info`). The other 1,162 have no such
facts and train with the runtime's seven.

The first 2,149 are personality conversations of 1 to 15 exchanges (5,751 Lu replies):

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
a pet name that appeared nine times.

The last 435 teach tool use with real tool calls. A hands-on chat showed the earlier model making
up times, writing "[time]", and saying it had checked when it hadn't; untouched Qwen3-4B called
tools more often but still reused old results from history and said "let me check" without
checking. The tool conversations:

- 90 time basics: time, date, weekday, "how long until", "am I late", in character.
- 70 repeated and challenged times: every new ask gets a fresh call even with an old result in
  history; "that's wrong" gets a fresh call, not a cave-in; "what did you say before?" is
  answered from history without a call.
- 70 machine info and errors: answers only from the result's fields; honest, in-character
  failures with no guessing, and a retry when asked.
- 80 without a tool: time trivia, history recall, requests for things Lu has no tool for
  (weather, lights, timers) with no invented result, and traps ("pretend you checked").
- 70 with extended tool lists (weather, timers, lights, music, volume, battery, search, head
  rotation) in shuffled order, so Lu reads the definitions instead of memorizing two names:
  correct arguments, asking when one is missing, two calls in a row, and no call when the needed
  tool isn't listed.
- 55 long chaotic chats (15 in child mode) where tools are a small part and Lu follows topic
  changes.

The last 625 teach the memory tools (`remember`, `recall`, `list_memories`, `update_memory`,
`forget`). In a hands-on chat the tool-round model said "I'll remember it" without saving, saved
nothing on its own, ignored what `recall` returned, invented a tool, mixed up facts in a long
chat (asking how a job was going before it had started), and would have saved "next Monday" as
said. The memory conversations (97 in child mode, 1,025 tool calls):

- 70 remember on request: many phrasings, several facts in one message, "did you save that?"
  answered from history, facts already saved, vague requests.
- 80 remember on its own: lasting facts saved mid-chat without being asked; 25 chats with only
  small talk, moods, and one-off remarks where nothing is saved; changed facts updated.
- 45 dates: relative dates ("next Monday", "in two weeks") get a `get_time` call and are saved
  as real dates ("on Monday, October 13, 2025"); countdowns from saved dates; date corrections.
- 65 recall: answers only from the result, plain "nothing on that" when empty, past
  conversations, partial results with no guessing.
- 75 listing and tool questions: "what do you know about me", counts past the newest twenty,
  an empty memory, "what can you do" in plain words, and 26 with shuffled extended tool lists
  (no list tool: search by name; no save tool: honest that it can't save).
- 50 corrections: `recall` then `update_memory`, or the id from an earlier save in the chat.
- 45 forgetting: `recall` then `forget`, nothing found, ambiguous matches, "forget everything".
- 75 "Things you remember": injected memories used with no call and no added detail, and
  irrelevant ones (pulled in by a shared word) ignored.
- 60 history accuracy: chats resumed after a restart where upcoming things stay upcoming, long
  detail-heavy chats, and the person as Lu's builder.
- 60 more: child-mode memory (no addresses, schools, or phone numbers saved), tool errors with
  honest replies and retries, and passwords, PINs, and card numbers declined.

After a memory change that worked, the reply starts with a fixed line, so the person can tell it
really happened: "Got it, I've saved that to my onboard memory." (`remember`), "Got it, I've
updated that in my onboard memory." (`update_memory`), "Got it, I've deleted that from my onboard
memory." (`forget`). No reply claims a change any other way. Saved text is a short third-person
sentence ("Mara's dog is named Biscuit.", or "The user ..." when no name is known). Every memory
conversation was replayed through the runtime's own `memory.py` (SQLite and FTS5), so its
injected memories, ids, and `recall`/`list_memories` results are exactly what the robot would
produce.

Spoken replies say times in words ("twenty past seven"), never mention tools by name, and never
say "let me check": the check is the tool call itself. `validate_data.py` enforces the message
order, at most three tool calls per user message (the runtime's agent loop limit), and the
confirmation lines, and rejects square-bracket placeholders in spoken replies.

Held-out test sets (never train on these):

- `data/eval_prompts.jsonl`: 56 everyday prompts, tagged by category (6 under `tools`;
  `compare.py` sends the runtime's tools, and a tool call shows up as `<tool_call>` text).
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

The 182 conversations marked `child` train with the same note, so the model has seen it. In
child mode Lu is also trained not to save a child's address, school, or phone number, and the
Lu-3 runtime starts a fresh memory session so nothing from before carries over. The detector is a
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

Memory, in the Lu-3 runtime (not training issues):

- `forget` deletes a saved memory, but the messages where the person said it stay in the chat
  archive, so `recall` can still turn the fact up.
- The automatic lookup matches whole words with no stemming, so "dogs" doesn't find a memory
  about a "dog", and an unrelated shared word ("work") can pull in an irrelevant memory. The
  training data covers ignoring those.
- The terminal shows `[tool] <name>` but not the arguments, so it isn't visible what `recall`
  searched for.

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
