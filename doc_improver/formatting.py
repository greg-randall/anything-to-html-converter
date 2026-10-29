"""Formatting content into Markdown with the LLM."""
from .llm import llm_request
from .prompts import fill, load_prompt

# Prompts for formatting, from config/prompts/format.yaml. The system prompt holds the rules: its {placeholder_rule}
# is replaced with FORMAT_PLACEHOLDER_RULE (or nothing). The user prompt holds the document: its {content} is
# replaced with it. In FORMAT_PLACEHOLDER_RULE, {prefix} is replaced with the image placeholder prefix.
FORMAT_SYSTEM_PROMPT = load_prompt("format", "system", slots=("placeholder_rule",))
FORMAT_USER_PROMPT = load_prompt("format", "user", slots=("content",))
FORMAT_PLACEHOLDER_RULE = load_prompt("format", "placeholder_rule", slots=("prefix",))


def format_content_with_llm(content, connection, model, max_tokens, effort=None, image_placeholder_prefix=None,
                            label="Formatting", verbose=False):
    """Formats content into Markdown with the model at connection (see llm.llm_request), handling raw text,
    existing Markdown, or CSV.

    If image_placeholder_prefix is given, the prompt tells the model to copy those image placeholders unchanged.
    label is shown on the progress line; verbose prints the call's connection and usage details.
    Returns (text, tokens): text is None if the call failed (the error is printed); tokens is a dict of
    input, output, cache_write and cache_read token counts (pricing.make_tokens), or None if no usage was
    received.
    """
    placeholder_rule = ""
    if image_placeholder_prefix:
        placeholder_rule = fill(FORMAT_PLACEHOLDER_RULE, prefix=image_placeholder_prefix)

    system_prompt = fill(FORMAT_SYSTEM_PROMPT, placeholder_rule=placeholder_rule)
    # The content is the only slot filled in the user prompt, so text in it that looks like a slot is never replaced
    user_prompt = fill(FORMAT_USER_PROMPT, content=content)
    return llm_request(connection, system_prompt, user_prompt, model=model, max_tokens=max_tokens,
                       effort=effort, label=label, verbose=verbose)
