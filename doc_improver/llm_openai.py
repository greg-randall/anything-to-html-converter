"""Requests in the OpenAI Chat Completions style: OpenAI, Azure OpenAI's v1 endpoint, or any OpenAI-compatible
server (Ollama, LM Studio, vLLM, OpenRouter, ...)."""
import openai
from tqdm import tqdm

from .pricing import make_tokens


def openai_tokens(usage):
    """Return the token counts of an OpenAI usage object as a dict (see pricing.make_tokens).

    prompt_tokens includes the cached tokens, so they are taken out of input and counted as cache_read.
    """
    details = usage.prompt_tokens_details
    cached = (details.cached_tokens if details is not None else None) or 0
    return make_tokens(input=usage.prompt_tokens - cached, output=usage.completion_tokens, cache_read=cached)


def openai_response_text(finish_reason, text, refusal):
    """Return (text, error) for a finished streamed response: the text, or None and an error message.

    Only a normal finish (finish_reason "stop") counts as success; being cut off at the token limit, a content
    filter, a refusal or a stream that ended without a finish reason is an error, so partial or missing
    output is never used.
    """
    if refusal:
        return None, f"The model refused the request: {refusal}"
    if finish_reason == "length":
        return None, ("The response was cut off at the max tokens limit and is incomplete. "
                      "Not returning partial output. Raise the max tokens setting.")
    if finish_reason == "content_filter":
        return None, "The response was blocked by the content filter."
    if finish_reason != "stop":
        return None, f"Unexpected finish reason {finish_reason!r}; not using the output."
    if not text.strip():
        return None, "The model finished normally but returned no text."
    return text, None


def openai_content(user_content):
    """Return the user message content in OpenAI's format: a string stays as it is; a list of parts
    (see llm.llm_request) becomes image_url parts (base64 data URLs) and text parts, in the same order."""
    if isinstance(user_content, str):
        return user_content
    parts = []
    for part in user_content:
        if part["type"] == "image":
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{part['media_type']};base64,{part['data']}"}})
        else:
            parts.append({"type": "text", "text": part["text"]})
    return parts


def openai_request(connection, system_prompt, user_content, *, model, max_tokens, effort=None, label="OpenAI",
                   verbose=False, json_schema=None, show_progress=True, log=print):
    """Send one streamed request to an OpenAI-style endpoint; see llm.llm_request for the arguments.

    max_tokens is sent as max_completion_tokens, effort as reasoning_effort, and json_schema as a strict
    json_schema response_format. No temperature is sent (reasoning models reject it). The token usage comes
    in the stream's last chunk (stream_options include_usage).

    Returns (text, tokens): tokens is a dict (pricing.make_tokens) with the cached prompt tokens counted as
    cache_read, or None if the server reported no usage.
    """
    client = openai.OpenAI(api_key=connection["api_key"], base_url=connection["base_url"])
    connection_line = (f"Using the OpenAI API at {client.base_url} (model {model}, "
                       f"effort {effort if effort else 'not set, API default'})")
    if verbose:
        log(connection_line)

    def print_failure_details(received=False, finish_reason=None, usage=None):
        """Print the call's details after a failure; with verbose they were already printed."""
        if verbose:
            return
        log(connection_line)
        if not received:
            log("No response was received, so there is no finish reason or token usage.")
        else:
            log(f"Finish reason: {finish_reason}")
            log(f"Full usage: {usage}")

    request_params = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": openai_content(user_content)}],
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if effort:
        request_params["reasoning_effort"] = effort
    if json_schema:
        request_params["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "response", "schema": json_schema,
                                                             "strict": True}}

    text_parts = []
    refusal_parts = []
    finish_reason = None
    usage = None
    response_model = None
    received = False
    try:
        with client.chat.completions.create(**request_params) as stream, \
                tqdm(desc=f"{label}: receiving", unit=" chars", leave=False, disable=not show_progress) as progress:
            for chunk in stream:
                received = True
                response_model = chunk.model or response_model
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    if choice.delta.content:
                        text_parts.append(choice.delta.content)
                        progress.update(len(choice.delta.content))
                    if choice.delta.refusal:
                        refusal_parts.append(choice.delta.refusal)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
    except openai.AuthenticationError as e:
        log(f"\n--- Authentication Error ---\nThe API key (LLM_API_KEY) was rejected.\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.PermissionDeniedError as e:
        log(f"\n--- Permission Denied ---\nThe key has no access to model {model}.\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.NotFoundError as e:
        log(f"\n--- Not Found ---\nCheck LLM_BASE_URL and the model name ({model}).\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.RateLimitError as e:
        log(f"\n--- Rate limited ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.BadRequestError as e:
        log(f"\n--- Bad request (rejected by the API) ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.APIStatusError as e:
        log(f"\n--- API error (HTTP {e.status_code}) ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.APIConnectionError as e:
        log(f"\n--- Connection error ---\nCould not reach {client.base_url}.\nDetails: {e}")
        print_failure_details()
        return None, None
    except openai.APIError as e:
        # e.g. an error event in the middle of the stream
        log(f"\n--- API error ---\nDetails: {e}")
        print_failure_details(received, finish_reason, usage)
        return None, None

    tokens = openai_tokens(usage) if usage is not None else None
    if verbose:
        log(f"Model: {response_model}  Finish reason: {finish_reason}")
        log(f"Full usage: {usage}")
    if usage is None:
        log(f"Note: {client.base_url} reported no token usage for this call, so its cost can't be estimated.")
    text, error = openai_response_text(finish_reason, "".join(text_parts), "".join(refusal_parts))
    if error:
        log(f"\n--- Error: no usable output ---\n{error}")
        print_failure_details(True, finish_reason, usage)
        return None, tokens
    return text, tokens
