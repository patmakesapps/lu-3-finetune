# Next round: memory tools

Notes from 2026-10-07 for the next training round. Nothing here is trained yet.

## Where things stand

- The model on Hugging Face is the tool-use round (`fda01fb`): `get_time` and `get_machine_info`.
- The runtime (Lu-3 repo, `software/lu3`) now has persistent memory in SQLite (`memory.py`) and
  four more tools: `remember`, `recall`, `update_memory`, `forget`. The model has never seen them.
- The runtime also reloads the last conversation's history on restart, and before each reply it
  adds memories matching the latest message to the end of the system prompt:

  ```
  <system prompt> Things you remember: <memory text> <memory text> ...
  ```

  (up to five; nothing is added when none match; the child note, if on, comes before it).

## Before training

1. Finish the robot's tool list first. Retrain once, after all the tools exist, so the
   definitions in training match the runtime.
2. Copy the exact definitions from `Lu-3/software/lu3/tools.py` (`TOOL_DEFINITIONS`) into
   `data/tools.json`, and add the new names to `tools` in `config.json`.
3. `train.py` only varies the system prompt by the `child` flag. It needs a per-example field
   (e.g. `"memories": ["..."]`) that appends ` Things you remember: ...` the way the runtime does.
4. Decide on runtime extras before training, since they change the tool list (see below).

## Hands-on test with the current model (2026-10-07)

A chat, a restart, then a second chat on the same history.

What worked:

- After the restart Lu greeted Pat by name and brought up the new job (from reloaded history).
- `get_time`: called when asked, time read out in words.
- `remember` worked once called, and Lu reported the id it got.
- In character the whole time, short spoken replies.

What failed:

1. **Said it saved without saving.** "remember that I make robots" got "I'll remember it: Pat
   makes robots" with no tool call. It only called `remember` after Pat said he didn't see it.
   This is the same "said it checked when it hadn't" failure the tool round fixed for time.
2. **Saved nothing on its own.** Pat mentioned a new job, its start (two work days left, a week
   off, then the new job), leaving at 4pm, two finished robots (one drives around the house, one
   sits on the desk), three models in progress with three of a kind being built, and that Lu
   isn't physically started yet. None of it was saved.
3. **Ignored the recall result.** Asked to use recall and report, it called `recall`, then
   listed its tools again and invented a tool ("the ability to keep you company").
4. **Counting memories.** "how many memories do you have stored?" called `recall` and said one.
   That was right, but likely from history: `recall` searches words, so no query lists them all.
5. **Mixed up facts in the conversation.** Asked how Pat was settling into the new job he hadn't
   started; said one robot was "already halfway built" (Pat said Lu isn't started); garbled the
   timeline ("Two full days after that, plus a week off!").
6. **"What tools can you see"** first got confusion, then a rough list that echoed the
   definition text ("remember things ... as a short sentence").
7. **Relative dates.** "I start next Monday" would go stale if saved as said.

## Data to add

- **Remember on request:** the `remember` call happens in the same turn. Never say "I'll remember"
  or "noted" without the call (same rule as "let me check").
- **Remember on its own:** lasting facts get saved without being asked: names, people, pets, jobs,
  schedules, projects, preferences, decisions. Small talk, moods, and one-off remarks don't.
  Usually one fact per call, as a short sentence about the person ("Pat makes robots.").
- **Dates:** relative dates ("next Monday", "in two weeks") get a `get_time` call first and are
  saved as real dates.
- **Recall:** answer only from what came back; an empty result is said plainly ("I don't have
  anything on that"); never invent. Use the ids from the result for updates and deletes.
- **Corrections:** "actually it's Tuesday" -> `recall` to find it, then `update_memory`. Not a
  second `remember`.
- **Forget:** on request, `recall` then `forget`, and confirm only after the result says it worked.
- **"Things you remember" in the system prompt:** use those facts naturally with no tool call;
  don't add details they don't contain. Include examples where they're irrelevant and ignored.
- **Resumed history:** greetings after a restart that use the history accurately (don't assume
  things moved on, like a job that hasn't started).
- **Tool questions:** "what tools do you have" answered from the definitions given, in plain
  words, without inventing any.
- **Staying accurate in long chats:** track what the person said (started vs. not started,
  built vs. planned) and correct course when told.
- **Child mode:** memory conversations in child mode too, with the same rules.
- Keep the earlier tool round's rules: tools never named in spoken replies, no "let me check".

## Runtime ideas (Lu-3 repo, not training)

- Print tool arguments next to `[tool] <name>` so it's visible what `recall` searched for.
- A `list_memories` tool for "what do you know about me?" and "how many memories?". Add it before
  training if wanted.
- The automatic lookup ORs every word of the message, so common words ("I", "the") will match
  most memories once there are many.
- Cosmetic: "Lu: " prints before `[tool]`, leaving the name on its own line.
