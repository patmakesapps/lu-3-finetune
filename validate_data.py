import argparse
import json
import re
from pathlib import Path

project_dir = Path(__file__).resolve().parent

config_path = project_dir / "config.json"
tools_path = project_dir / "data" / "tools.json"

parser = argparse.ArgumentParser(description="Check the training data format.")
parser.add_argument("--data", default=str(project_dir / "data" / "train.jsonl"))
data_path = Path(parser.parse_args().data)

with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

with tools_path.open(encoding="utf-8") as file:
    tool_library = json.load(file)

for name, definition in tool_library.items():
    if definition["function"]["name"] != name:
        raise ValueError(f"tools.json: entry '{name}' defines a tool named '{definition['function']['name']}'.")

# Placeholders like [time] are exactly what the model wrote instead of calling
# a tool, so no spoken reply may contain one, and none may hold tool-call markup.
placeholder = re.compile(r"\[[^\]]*\]|<tool_call>|</tool_call>|<tool_response>")

# The runtime's agent loop makes at most four requests per user message, so
# three tool calls is the most that can still end in a spoken reply.
max_calls_per_turn = 3

# After a memory change that worked, the spoken reply says so with the tool's
# fixed line, and no reply says it without one: the line is how the person
# knows the change really happened.
confirmations = {
    "remember": "Got it, I've saved that to my onboard memory.",
    "update_memory": "Got it, I've updated that in my onboard memory.",
    "forget": "Got it, I've deleted that from my onboard memory.",
}


def memory_change_worked(name, result):
    if name == "remember":
        return "id" in result and "error" not in result
    if name == "update_memory":
        return result.get("updated") is True
    if name == "forget":
        return result.get("forgotten") is True
    return False

conversation_count = 0
reply_count = 0
tool_call_count = 0


def check_tool_call(line_number, message, tool_names):
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        # The runtime sends parallel_tool_calls: false, so one call per turn.
        raise ValueError(f"Line {line_number}: a tool-call turn must have exactly one tool call.")
    if message.get("content") != "":
        raise ValueError(f"Line {line_number}: a tool-call turn must have empty content.")

    call = tool_calls[0]
    function = call.get("function") or {}
    if call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"]:
        raise ValueError(f"Line {line_number}: tool call needs an id and type 'function'.")
    if function.get("name") not in tool_names:
        raise ValueError(f"Line {line_number}: tool '{function.get('name')}' is not in this conversation's tools.")
    if not isinstance(function.get("arguments"), str) or not isinstance(json.loads(function["arguments"]), dict):
        raise ValueError(f"Line {line_number}: tool arguments must be a JSON object string.")

    return call["id"], function["name"]


with data_path.open(encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        conversation = json.loads(line)
        messages = conversation.get("messages")

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {line_number}: messages must be a nonempty list.")

        tool_names = conversation.get("tools", config["tools"])
        if not isinstance(tool_names, list) or any(name not in tool_library for name in tool_names):
            raise ValueError(f"Line {line_number}: tools must be a list of names from data/tools.json.")

        if "child" in conversation and not isinstance(conversation["child"], bool):
            raise ValueError(f"Line {line_number}: child must be true or false.")

        # Allowed order: user, then either a spoken reply or a run of
        # tool call -> tool result pairs that ends in a spoken reply.
        previous = None
        pending_call_id = None
        pending_call_name = None
        calls_this_turn = 0
        changes_this_turn = set()

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Line {line_number}: each message must be an object.")

            role = message.get("role")

            if role == "user":
                if previous not in (None, "reply"):
                    raise ValueError(f"Line {line_number}: a user message must start the conversation or follow a spoken reply.")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Line {line_number}: message content must be nonempty text.")
                memories = message.get("memories", [])
                if not isinstance(memories, list) or not all(isinstance(m, str) and m.strip() for m in memories):
                    raise ValueError(f"Line {line_number}: memories must be a list of nonempty strings.")
                calls_this_turn = 0
                changes_this_turn = set()
                previous = "user"

            elif role == "assistant" and "tool_calls" in message:
                if previous not in ("user", "tool"):
                    raise ValueError(f"Line {line_number}: a tool call must follow a user message or a tool result.")
                pending_call_id, pending_call_name = check_tool_call(line_number, message, tool_names)
                tool_call_count += 1
                calls_this_turn += 1
                if calls_this_turn > max_calls_per_turn:
                    raise ValueError(f"Line {line_number}: more than {max_calls_per_turn} tool calls for one user message.")
                previous = "tool_call"

            elif role == "assistant":
                if previous not in ("user", "tool"):
                    raise ValueError(f"Line {line_number}: a spoken reply must follow a user message or a tool result.")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"Line {line_number}: message content must be nonempty text.")
                if placeholder.search(content):
                    raise ValueError(f"Line {line_number}: spoken reply contains a placeholder or tool markup: {content!r}")
                said = {name for name, line in confirmations.items() if line in content}
                if said != changes_this_turn or (said and not content.startswith(tuple(confirmations.values()))):
                    raise ValueError(f"Line {line_number}: a reply must start with the confirmation line of each "
                                     f"memory change that worked this turn, and only those: {content!r}")
                reply_count += 1
                previous = "reply"

            elif role == "tool":
                if previous != "tool_call":
                    raise ValueError(f"Line {line_number}: a tool result must follow a tool call.")
                if message.get("tool_call_id") != pending_call_id:
                    raise ValueError(f"Line {line_number}: tool result id does not match the tool call.")
                if not isinstance(message.get("content"), str) or not isinstance(json.loads(message["content"]), dict):
                    raise ValueError(f"Line {line_number}: tool result content must be a JSON object string.")
                if memory_change_worked(pending_call_name, json.loads(message["content"])):
                    changes_this_turn.add(pending_call_name)
                previous = "tool"

            else:
                raise ValueError(f"Line {line_number}: unknown role '{role}'.")

        if previous != "reply":
            raise ValueError(f"Line {line_number}: conversation must end with a spoken assistant reply.")

        conversation_count += 1

if conversation_count == 0:
    raise ValueError("The training file contains no conversations.")

print(f"Base model: {config['base_model']}")
print(f"Valid conversations: {conversation_count}")
print(f"Spoken replies: {reply_count}")
print(f"Tool calls: {tool_call_count}")
