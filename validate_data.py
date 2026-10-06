import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent

config_path = project_dir / "config.json"
data_path = project_dir / "data" / "train.jsonl"

with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

conversation_count = 0
reply_count = 0

with data_path.open(encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        conversation = json.loads(line)
        messages = conversation.get("messages")

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {line_number}: messages must be a nonempty list.")

        if len(messages) % 2 != 0:
            raise ValueError(f"Line {line_number}: conversation must end with an assistant reply.")

        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"

            if not isinstance(message, dict):
                raise ValueError(f"Line {line_number}: each message must be an object.")

            if message.get("role") != expected_role:
                raise ValueError(f"Line {line_number}: expected role '{expected_role}'.")

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Line {line_number}: message content must be nonempty text.")

            if expected_role == "assistant":
                reply_count += 1

        conversation_count += 1

if conversation_count == 0:
    raise ValueError("The training file contains no conversations.")

print(f"Base model: {config['base_model']}")
print(f"Valid conversations: {conversation_count}")
print(f"Assistant replies: {reply_count}")