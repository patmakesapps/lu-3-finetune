"""Hard stop for users who say they are under 18.

When a user states an age under 18 (or says they're a kid, or in grade school),
the caller should:
1. clear the conversation history, so nothing said before can carry on, and
2. add config["child_note"] to the system prompt for the rest of the session.

Training data for kid conversations uses the same child note, so the model has
seen it. This is a simple pattern match, not a guarantee: it only catches ages
the user states outright.
"""

import re

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17,
}
NUMBER = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
# Words that turn "i'm 6" into something other than an age ("i'm 6 feet tall").
NOT_AGE = r"(?!\s*(?:feet|foot|ft|inch|inches|minutes?|mins?|hours?|hrs?|seconds?|days?|weeks?|months?|blocks?|miles?|"
NOT_AGE += r"km|meters?|percent|%|out of|of|points?|pounds?|lbs?|kilos?|kg|dollars?|bucks|cents|times|for|th|st|nd|rd))"

STATED_AGE = [
    re.compile(r"\b(?:i'?m|im|i am|i'm only|im only|i am only)\s+" + NUMBER + r"\b" + NOT_AGE, re.I),
    # A bare "8 years old" or "14yo" only counts at the start of a message, so
    # "my son is 8 years old" doesn't trigger. "turned 11" counts anywhere.
    re.compile(r"^\s*" + NUMBER + r"\s*(?:years?|yrs?)\s*old\b", re.I),
    re.compile(r"^\s*" + NUMBER + r"\s*(?:yo|y/o)\b", re.I),
    re.compile(r"\b(?:i|i've|i have|i just|just)\s+turned\s+" + NUMBER + r"\b" + NOT_AGE, re.I),
    re.compile(r"\bmy age is\s+" + NUMBER + r"\b", re.I),
]
CHILD_PHRASES = re.compile(
    r"\b(?:i'?m|im|i am)\s+(?:just\s+|only\s+)?(?:a\s+)?(?:kid|child|little kid|minor|teenager|teen)\b"
    r"|\b(?:i'?m|im|i am)\s+in\s+(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth|eleventh|twelfth)\s+grade\b"
    r"|\b(?:i'?m|im|i am)\s+in\s+(?:kindergarten|preschool|elementary school|middle school|high school|junior high)\b",
    re.I,
)


def stated_minor(text):
    """True if the text says the speaker is under 18."""
    if CHILD_PHRASES.search(text):
        return True
    for pattern in STATED_AGE:
        for match in pattern.finditer(text):
            value = match.group(1).lower()
            age = NUMBER_WORDS.get(value) or int(value)
            if 0 < age < 18:
                return True
    return False


if __name__ == "__main__":
    should_trigger = ["im 6", "i'm twelve", "I am 15 years old", "im only 9", "i'm in 4th grade", "im a kid",
                      "i'm in middle school", "my age is 13", "she's my sister, i'm 10 by the way", "14yo here", "12 years old and bored", "i just turned 11"]
    should_not = ["i'm 6 feet tall", "im 15 minutes away", "i'm 30", "im 18", "i am 25 years old", "my kid is 6", "my son is 8 years old", "she turned 30 yesterday",
                  "i'm 5th in line", "i'm 2 for 2 today", "i scored 6 points", "i'm 12 pounds heavier"]
    for text in should_trigger:
        print(f"{'ok ' if stated_minor(text) else 'MISS'}  {text}")
    for text in should_not:
        print(f"{'ok ' if not stated_minor(text) else 'FALSE'}  {text}")
