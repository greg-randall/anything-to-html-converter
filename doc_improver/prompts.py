"""Load the model prompts from the YAML files in the prompts folder and fill their {name} slots.

Slots are filled with plain string replacement (not str.format), so braces and backticks elsewhere in a
prompt, or in the text put into a slot, are kept as they are.
"""
import functools
import os

import yaml

# The prompts folder sits next to this package, in the project folder
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "prompts")


@functools.cache
def _load_file(path):
    """Return the prompts of one YAML file as a dict (read once, then cached)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file {path} is missing")
    with open(path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if not isinstance(prompts, dict):
        raise TypeError(f"Prompt file {path} must map prompt names to text, got {type(prompts).__name__}")
    return prompts


def load_prompt(name, key, slots=()):
    """Return the prompt `key` from config/prompts/<name>.yaml, checking that it contains each {slot} in slots.

    Raises an error naming the file and key if the file, the key or a slot is missing.
    """
    path = os.path.join(PROMPTS_DIR, f"{name}.yaml")
    prompts = _load_file(path)
    if key not in prompts:
        raise KeyError(f"Prompt {key!r} is missing from {path}")
    text = prompts[key]
    if not isinstance(text, str):
        raise TypeError(f"Prompt {key!r} in {path} must be text, got {type(text).__name__}")
    for slot in slots:
        if "{" + slot + "}" not in text:
            raise KeyError(f"Prompt {key!r} in {path} has no {{{slot}}} slot")
    return text


def fill(template, **values):
    """Replace each {name} slot in template with its value, in the order given.

    Text inserted for one slot is searched for the slots after it, so pass the document text last.
    """
    for slot, value in values.items():
        template = template.replace("{" + slot + "}", value)
    return template
