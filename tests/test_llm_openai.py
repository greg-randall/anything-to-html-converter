"""openai_request with a fake OpenAI client that streams fake chunks: nothing is sent anywhere."""
from types import SimpleNamespace

import httpx
import openai

CONNECTION_SETTINGS = {'api': 'openai', 'base_url': 'https://fake.example.invalid/v1/', 'api_key': 'fake-key'}
CONNECTION = "Using the OpenAI API at https://fake.example.invalid/v1/ (model fake-model, effort low)"


class FakeStream:
    def __init__(self, chunks, error_after=None):
        self.chunks = chunks
        self.error_after = error_after

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        yield from self.chunks
        if self.error_after:
            raise self.error_after


def fake_openai(monkeypatch, create):
    """Replace openai.OpenAI with a fake whose chat.completions.create is create(**params)."""

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            assert (api_key, base_url) == ('fake-key', 'https://fake.example.invalid/v1/')
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)


def chunk(content=None, finish_reason=None, refusal=None, usage=None, choices=True):
    delta = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(model='fake-model-2026', usage=usage,
                           choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)] if choices else [])


USAGE = SimpleNamespace(prompt_tokens=30, completion_tokens=900, prompt_tokens_details=None)
USAGE_TEXT = "namespace(prompt_tokens=30, completion_tokens=900, prompt_tokens_details=None)"
TOKENS = {'input': 30, 'output': 900, 'cache_write': 0, 'cache_read': 0}


def streamed(*texts, finish_reason='stop', usage=USAGE, refusal=None):
    """The chunks of a response: one per text, a last one with the finish reason, then the usage-only chunk."""
    chunks = [chunk(content=text) for text in texts]
    chunks.append(chunk(finish_reason=finish_reason, refusal=refusal))
    if usage is not None:
        chunks.append(chunk(usage=usage, choices=False))
    return chunks


def capture(monkeypatch, chunks):
    sent = {}

    def create(**params):
        sent.update(params)
        return FakeStream(chunks)

    fake_openai(monkeypatch, create)
    return sent


def request(api, content="user", **options):
    return api.openai_request(CONNECTION_SETTINGS, "system", content, model='fake-model', max_tokens=900,
                              effort='low', label="Test", **options)


def test_streamed_text_is_joined_and_nothing_printed_by_default(api, monkeypatch, capsys):
    capture(monkeypatch, streamed('Hel', 'lo'))
    assert request(api) == ('Hello', TOKENS)
    assert capsys.readouterr().out == ""


def test_verbose_prints_the_details(api, monkeypatch, capsys):
    capture(monkeypatch, streamed('Hello'))
    assert request(api, verbose=True) == ('Hello', TOKENS)
    assert capsys.readouterr().out == (
        f"{CONNECTION}\n"
        "Model: fake-model-2026  Finish reason: stop\n"
        f"Full usage: {USAGE_TEXT}\n")


def test_request_parameters(api, monkeypatch):
    sent = capture(monkeypatch, streamed('Hello'))
    request(api)
    assert sent == {'model': 'fake-model',
                    'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'}],
                    'max_completion_tokens': 900, 'stream': True, 'stream_options': {'include_usage': True},
                    'reasoning_effort': 'low'}


def test_no_effort_sends_no_reasoning_effort(api, monkeypatch):
    sent = capture(monkeypatch, streamed('Hello'))
    api.openai_request(CONNECTION_SETTINGS, "system", "user", model='fake-model', max_tokens=900)
    assert 'reasoning_effort' not in sent
    assert 'temperature' not in sent


def test_json_schema_is_sent_as_a_strict_response_format(api, monkeypatch):
    sent = capture(monkeypatch, streamed('{"alt_text": ', '"Hi"}'))
    schema = {"type": "object", "properties": {"alt_text": {"type": "string"}}, "required": ["alt_text"],
              "additionalProperties": False}
    assert request(api, json_schema=schema) == ('{"alt_text": "Hi"}', TOKENS)
    assert sent['response_format'] == {"type": "json_schema",
                                       "json_schema": {"name": "response", "schema": schema, "strict": True}}


def test_image_and_text_parts_become_openai_parts(api, monkeypatch):
    content = [api.image_part('image/jpeg', 'QUJD'), api.text_part('Describe it.')]
    expected = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,QUJD'}},
                {'type': 'text', 'text': 'Describe it.'}]
    assert api.openai_content(content) == expected
    assert api.openai_content('plain') == 'plain'
    sent = capture(monkeypatch, streamed('A logo'))
    request(api, content=content)
    assert sent['messages'][1] == {'role': 'user', 'content': expected}


def test_llm_request_sends_openai_style_through_openai_request(api, monkeypatch):
    sent = capture(monkeypatch, streamed('Hello'))
    assert api.llm_request(CONNECTION_SETTINGS, "system", "user", model='fake-model', max_tokens=900) == (
        'Hello', TOKENS)
    assert sent['max_completion_tokens'] == 900


def test_cut_off_response_is_an_error_with_the_details(api, monkeypatch, capsys):
    capture(monkeypatch, streamed('partial', finish_reason='length'))
    assert request(api) == (None, TOKENS)
    assert capsys.readouterr().out == (
        "\n--- Error: no usable output ---\n"
        "The response was cut off at the max tokens limit and is incomplete. "
        "Not returning partial output. Raise the max tokens setting.\n"
        f"{CONNECTION}\n"
        "Finish reason: length\n"
        f"Full usage: {USAGE_TEXT}\n")


def test_response_checks(api):
    assert api.openai_response_text('stop', 'Hi', '') == ('Hi', None)
    assert api.openai_response_text('stop', 'Hi', 'I cannot help with that.') == (
        None, "The model refused the request: I cannot help with that.")
    assert api.openai_response_text('content_filter', '', '') == (
        None, "The response was blocked by the content filter.")
    assert api.openai_response_text(None, 'Hi', '') == (
        None, "Unexpected finish reason None; not using the output.")
    assert api.openai_response_text('stop', ' \n', '') == (None, "The model finished normally but returned no text.")


def test_refusal_is_an_error(api, monkeypatch, capsys):
    capture(monkeypatch, streamed(refusal="I can't help with that."))
    assert request(api) == (None, TOKENS)
    assert "The model refused the request: I can't help with that." in capsys.readouterr().out


def test_missing_usage_is_reported_and_tokens_are_none(api, monkeypatch, capsys):
    capture(monkeypatch, streamed('Hello', usage=None))
    assert request(api) == ('Hello', None)
    assert capsys.readouterr().out == (
        "Note: https://fake.example.invalid/v1/ reported no token usage for this call, so its cost can't be "
        "estimated.\n")


def test_connection_error_prints_endpoint_and_model(api, monkeypatch, capsys):
    def create(**params):
        raise openai.APIConnectionError(request=httpx.Request('POST', 'https://fake.example.invalid/v1/'))

    fake_openai(monkeypatch, create)
    assert request(api) == (None, None)
    assert capsys.readouterr().out == (
        "\n--- Connection error ---\n"
        "Could not reach https://fake.example.invalid/v1/.\n"
        "Details: Connection error.\n"
        f"{CONNECTION}\n"
        "No response was received, so there is no finish reason or token usage.\n")


def test_status_error_names_the_setting_to_check(api, monkeypatch, capsys):
    response = httpx.Response(404, request=httpx.Request('POST', 'https://fake.example.invalid/v1/'))

    def create(**params):
        raise openai.NotFoundError("Resource not found", response=response, body=None)

    fake_openai(monkeypatch, create)
    assert request(api) == (None, None)
    assert "\n--- Not Found ---\nCheck LLM_BASE_URL and the model name (fake-model).\n" in capsys.readouterr().out


def test_error_in_the_middle_of_the_stream(api, monkeypatch, capsys):
    error = openai.APIError("stream broke", request=httpx.Request('POST', 'https://fake.example.invalid/v1/'),
                            body=None)
    fake_openai(monkeypatch, lambda **params: FakeStream([chunk(content='part')], error_after=error))
    assert request(api) == (None, None)
    out = capsys.readouterr().out
    assert "\n--- API error ---\nDetails: stream broke\n" in out
    assert "Finish reason: None\n" in out


def test_parallel_calls_can_turn_off_the_bar_and_collect_their_lines(api, monkeypatch, capsys):
    capture(monkeypatch, streamed('partial', finish_reason='length'))
    lines = []
    assert request(api, show_progress=False, log=lines.append) == (None, TOKENS)
    assert capsys.readouterr() == ("", "")
    assert "The response was cut off at the max tokens limit" in "\n".join(lines)


def test_openai_tokens_moves_cached_tokens_out_of_input(api):
    usage = SimpleNamespace(prompt_tokens=5000, completion_tokens=80,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=4096))
    assert api.openai_tokens(usage) == {'input': 904, 'output': 80, 'cache_write': 0, 'cache_read': 4096}
    usage = SimpleNamespace(prompt_tokens=5000, completion_tokens=80,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=None))
    assert api.openai_tokens(usage) == {'input': 5000, 'output': 80, 'cache_write': 0, 'cache_read': 0}


def test_openai_field_names_exist_in_the_installed_sdk():
    from openai.types.chat import chat_completion_chunk as chunk_types
    from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
    assert {'prompt_tokens', 'completion_tokens', 'prompt_tokens_details'} <= set(CompletionUsage.model_fields)
    assert 'cached_tokens' in PromptTokensDetails.model_fields
    assert {'model', 'choices', 'usage'} <= set(chunk_types.ChatCompletionChunk.model_fields)
    assert {'delta', 'finish_reason'} <= set(chunk_types.Choice.model_fields)
    assert {'content', 'refusal'} <= set(chunk_types.ChoiceDelta.model_fields)
