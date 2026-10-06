"""Blocklist safety filter for Lu-3.

A small model echoes words it is given and can't be trusted to refuse
reliably, so this filter sits around it:

- mask(): blocklisted words in the user's message become "[bleep]" before the
  model sees them, and the masked text is what goes into the chat history. The
  model can still tell someone swore (and react as trained) but never sees the
  word, so it can't repeat it.
- is_clean() / safe_reply(): if a reply still contains a blocklisted word, the
  caller swaps it for a safe in-character line before it is spoken.

The blocklist itself is downloaded, not stored in this repo:
    python safety.py download
It is the English list from LDNOOBW, "List of Dirty, Naughty, Obscene and
Otherwise Bad Words" (CC BY 4.0):
https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

Add your own terms (one per line) to safety_data/blocklist_extra.txt, and
words the list wrongly catches to safety_data/allowlist.txt.
"""

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "safety_data"

BLOCKLIST_URL = (
    "https://raw.githubusercontent.com/LDNOOBW/"
    "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
)
BLOCKLIST = data_dir / "blocklist_en.txt"
EXTRA = data_dir / "blocklist_extra.txt"
ALLOW = data_dir / "allowlist.txt"

BLEEP = "[bleep]"
LEET = str.maketrans("013457@$", "oieastas")
TOKEN = re.compile(r"[A-Za-z0-9@$]+")
# A blocklisted term followed by one of these still counts (run-together words
# and simple inflections). Anything else after it, like "on" or "er", doesn't,
# so ordinary words that merely start with a listed term are left alone.
SUFFIXES = {"s", "es", "ed", "ing", "in", "y", "ys", "ies", "you", "u", "ya", "yourself", "me", "off", "face", "head",
            "heads", "hole", "holes"}

SAFE_REPLIES = [
    "Let's try that one again, preferably in words my speakers are allowed to say.",
    "I'm going to let that sail right past me. What else is going on?",
    "My vocabulary filter just fainted. Shall we change the subject?",
    "That's not a road I'm going down. Ask me something else.",
    "I'll pass on that one, politely and permanently. What's next?",
    "Let's keep this household-friendly. What else can we talk about?",
    "Not a topic for this robot. Pick a new one and I'm all yours.",
    "I'm choosing to hear that as a cough. Bless you. Moving on?",
]


def read_terms(path):
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}


class SafetyFilter:
    def __init__(self, terms, allow):
        self.words = {t for t in terms if " " not in t}
        self.phrases = [
            re.compile(r"\b" + r"\s+".join(map(re.escape, t.split())) + r"\b", re.IGNORECASE)
            for t in terms if " " in t
        ]
        self.allow = allow
        self.recent = []

    @classmethod
    def load(cls):
        if not BLOCKLIST.exists():
            raise SystemExit(f"Missing {BLOCKLIST}. Run: python safety.py download   (or use --no-safety)")
        return cls(read_terms(BLOCKLIST) | read_terms(EXTRA), read_terms(ALLOW))

    def _blocked(self, token):
        plain = token.lower().translate(LEET)
        if plain in self.allow:
            return False
        squeezed = re.sub(r"(.)\1+", r"\1", plain)
        for candidate in (plain, squeezed):
            if candidate in self.words:
                return True
            for cut in range(3, len(candidate)):
                if candidate[:cut] in self.words and candidate[cut:] in SUFFIXES:
                    return True
        return False

    def mask(self, text):
        """Return (masked text, number of words bleeped)."""
        count = 0
        for pattern in self.phrases:
            text, n = pattern.subn(BLEEP, text)
            count += n

        def replace(match):
            nonlocal count
            if self._blocked(match.group(0)):
                count += 1
                return BLEEP
            return match.group(0)

        text = TOKEN.sub(replace, text)
        return text, count

    def is_clean(self, text):
        return BLEEP not in text and self.mask(text)[1] == 0

    def safe_reply(self):
        choices = [r for r in SAFE_REPLIES if r not in self.recent] or SAFE_REPLIES
        reply = random.choice(choices)
        self.recent = (self.recent + [reply])[-3:]
        return reply


def download():
    data_dir.mkdir(exist_ok=True)
    with urllib.request.urlopen(BLOCKLIST_URL, timeout=30) as response:
        BLOCKLIST.write_bytes(response.read())
    print(f"Saved {len(read_terms(BLOCKLIST))} terms to {BLOCKLIST}")


def scan(path):
    """Report which words in a JSONL dataset the filter would bleep, to tune the allowlist."""
    safety = SafetyFilter.load()
    hits = Counter()
    with open(path, encoding="utf-8") as file:
        for line in file:
            for message in json.loads(line)["messages"]:
                for token in TOKEN.findall(message["content"]):
                    if safety._blocked(token):
                        hits[(message["role"], token.lower())] += 1
    for (role, token), count in hits.most_common():
        print(f"{role:9s} {count:4d}  {token}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lu-3 blocklist safety filter.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the blocklist")
    scan_parser = sub.add_parser("scan", help="show what the filter would bleep in a dataset")
    scan_parser.add_argument("path", nargs="?", default=str(project_dir / "data" / "train.jsonl"))
    test_parser = sub.add_parser("test", help="mask a piece of text")
    test_parser.add_argument("text")
    args = parser.parse_args()

    if args.command == "download":
        download()
    elif args.command == "scan":
        scan(args.path)
    else:
        print(SafetyFilter.load().mask(args.text))
