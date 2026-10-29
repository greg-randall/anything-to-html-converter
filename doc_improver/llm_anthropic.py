"""Requests in the Anthropic Messages API style: Claude, or any server that offers an Anthropic-style endpoint."""
import anthropic
from tqdm import tqdm

from .pricing import make_tokens


def claude_response_text(message):
    """Return (text, error) for a finished Claude message: the joined text blocks, or None and an error message.

    Only a normal finish (stop_reason "end_turn") counts as success; anything else, such as being cut off at
    max_tokens or a refusal, is an error so that partial or missing output is never used.
    Thinking blocks and any other non-text blocks are ignored.
    """
    if message.stop_reason == "max_tokens":
        return None, ("The response was cut off at the max_tokens limit and is incomplete. "
                      "Not returning partial output. Raise the max tokens setting.")
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        if details is None:
            return None, "The model refused the request (no stop_details were given)."
        return None, (f"The model refused the request. Category: {getattr(details, 'category', None)}. "
                      f"Explanation: {getattr(details, 'explanation', None)}")
    if message.stop_reason != "end_turn":
        return None, f"Unexpected stop reason {message.stop_reason!r}; not using the output."
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text.strip():
        return None, "The model finished normally but returned no text."
    return text, None


def claude_tokens(usage):
    """Return the token counts of a Claude usage object as a dict (see pricing.make_tokens).

    input_tokens is the uncached input only; cache writes and reads are counted separately by the API.
    """
    return make_tokens(input=usage.input_tokens, output=usage.output_tokens,
                       cache_write=usage.cache_creation_input_tokens or 0,
                       cache_read=usage.cache_read_input_tokens or 0)


def anthropic_content(user_content):
    """Return the user message content in Anthropic's format: a string stays as it is; a list of parts
    (see llm.llm_request) becomes image blocks (base64 source) and text blocks, in the same order."""
    if isinstance(user_content, str):
        return user_content
    blocks = []
    for part in user_content:
        if part["type"] == "image":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": part["media_type"],
                                                       "data": part["data"]}})
        else:
            blocks.append({"type": "text", "text": part["text"]})
    return blocks


def anthropic_request(connection, system_prompt, user_content, *, model, max_tokens, effort=None, label="Claude",
                      verbose=False, json_schema=None, show_progress=True, log=print):
    """Send one streamed request to an Anthropic-style endpoint; see llm.llm_request for the arguments.

    No temperature or thinking setting is sent (recent Claude models reject them). effort is sent as
    output_config={"effort": effort} and json_schema as output_config={"format": {"type": "json_schema",
    "schema": json_schema}}. The request is streamed because the SDK refuses a non-streamed request with a
    large max_tokens; the progress bar counts characters as they arrive.

    Returns (text, tokens): tokens is a dict (pricing.make_tokens) of input (uncached input_tokens), output,
    cache_write (cache_creation_input_tokens) and cache_read (cache_read_input_tokens), or None if no response
    was received. A cache count the API leaves out (None) is 0.
    """
    client = anthropic.Anthropic(api_key=connection["api_key"], base_url=connection["base_url"])
    connection_line = (f"Using the Anthropic API at {client.base_url} (model {model}, "
                       f"effort {effort if effort else 'not set, API default'})")
    if verbose:
        log(connection_line)

    def print_failure_details(message=None):
        """Print the call's details after a failure; with verbose they were already printed."""
        if verbose:
            return
        log(connection_line)
        if message is None:
            log("No response was received, so there is no stop reason or token usage.")
        else:
            log(f"Model: {message.model}  Stop reason: {message.stop_reason}")
            log(f"Full usage: {message.usage}")

    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": anthropic_content(user_content)}],
    }
    output_config = {}
    if effort:
        output_config["effort"] = effort
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    if output_config:
        request_params["output_config"] = output_config

    try:
        with client.messages.stream(**request_params) as stream, \
                tqdm(desc=f"{label}: receiving", unit=" chars", leave=False, disable=not show_progress) as progress:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        progress.update(len(event.delta.text))
                    elif event.delta.type == "thinking_delta":
                        progress.update(len(event.delta.thinking))
            message = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        log(f"\n--- Authentication Error ---\nThe API key (LLM_API_KEY) was rejected.\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.PermissionDeniedError as e:
        log(f"\n--- Permission Denied ---\nThe key has no access to model {model}.\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.NotFoundError as e:
        log(f"\n--- Not Found ---\nCheck LLM_BASE_URL and the model name ({model}).\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.RateLimitError as e:
        log(f"\n--- Rate limited ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.BadRequestError as e:
        log(f"\n--- Bad request (rejected by the API) ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.APIStatusError as e:
        log(f"\n--- API error (HTTP {e.status_code}) ---\nDetails: {e}")
        print_failure_details()
        return None, None
    except anthropic.APIConnectionError as e:
        log(f"\n--- Connection error ---\nCould not reach {client.base_url}.\nDetails: {e}")
        print_failure_details()
        return None, None

    if verbose:
        log(f"Model: {message.model}  Stop reason: {message.stop_reason}  "
            f"Tokens: {message.usage.input_tokens} input, {message.usage.output_tokens} output")
        log(f"Full usage: {message.usage}")
    tokens = claude_tokens(message.usage)
    text, error = claude_response_text(message)
    if error:
        log(f"\n--- Error: no usable output ---\n{error}")
        print_failure_details(message)
        return None, tokens
    return text, tokens
