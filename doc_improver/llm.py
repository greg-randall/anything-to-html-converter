"""One way to send a request to the model, whichever API style the endpoint speaks."""
from .llm_anthropic import anthropic_request
from .llm_openai import openai_request

# The API styles an endpoint can speak, and the function that sends a request in that style
API_STYLES = {
    'anthropic': anthropic_request,
    'openai': openai_request,
}


def text_part(text):
    """A text part of a user message (see llm_request)."""
    return {"type": "text", "text": text}


def image_part(media_type, data):
    """An image part of a user message (see llm_request): data is the image file's bytes, base64-encoded as
    a str."""
    return {"type": "image", "media_type": media_type, "data": data}


def llm_request(connection, system_prompt, user_content, *, model, max_tokens, effort=None, label="Model",
                verbose=False, json_schema=None, show_progress=True, log=print):
    """Send one request and return (text, tokens).

    Args:
        connection (dict): api ('anthropic' or 'openai'), base_url (None for the API's official endpoint) and
            api_key, from config.get_connection.
        system_prompt (str): The instructions for the model.
        user_content (str or list): The user message: a string, or a list of parts made with text_part and
            image_part, sent in that order.
        model (str): The model (or deployment) name.
        max_tokens (int): The maximum number of output tokens (thinking included).
        effort (str, optional): The effort level (Anthropic effort, OpenAI reasoning_effort); if None, the API
            default applies. A model that doesn't support it rejects the request.
        label (str): Shown on the progress bar.
        verbose (bool): Print the endpoint, model, stop reason and full usage for every call. Without it these
            are printed only when the call fails.
        json_schema (dict, optional): If given, the reply must be JSON matching this schema (structured output).
            The JSON text is returned as-is; the caller parses it.
        show_progress (bool): Show the character counter while the response streams in. Turn it off for calls
            that run in parallel, so several bars don't overwrite each other on screen.
        log (callable): Called with each line this function would print (default: print). Calls that run in
            parallel pass a collector, so their lines can be printed later in order instead of interleaved.

    Returns:
        tuple: (text, tokens). text is the response text, or None if the call failed (the error is printed
        through log). tokens is a dict of input, output, cache_write and cache_read token counts
        (pricing.make_tokens), or None if no usage was received.
    """
    request = API_STYLES[connection["api"]]
    return request(connection, system_prompt, user_content, model=model, max_tokens=max_tokens, effort=effort,
                   label=label, verbose=verbose, json_schema=json_schema, show_progress=show_progress, log=log)
