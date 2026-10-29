"""claude_response_text and anthropic_request with fake Claude messages and a fake client (no API calls)."""
from types import SimpleNamespace

import anthropic
import httpx


def block(kind, text=None):
    return SimpleNamespace(type=kind, text=text)


def message(stop_reason, content=(), **extra):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), **extra)


def test_text_blocks_are_joined_and_other_blocks_ignored(api):
    msg = message('end_turn', [block('thinking'), block('text', 'Hello '), block('text', 'world')])
    assert api.claude_response_text(msg) == ('Hello world', None)


def test_cut_off_at_max_tokens(api):
    assert api.claude_response_text(message('max_tokens', [block('text', 'partial')])) == (
        None, ("The response was cut off at the max_tokens limit and is incomplete. "
               "Not returning partial output. Raise the max tokens setting."))


def test_refusal_with_and_without_stop_details(api):
    details = SimpleNamespace(category='cyber', explanation='Not allowed.')
    assert api.claude_response_text(message('refusal', stop_details=details)) == (
        None, "The model refused the request. Category: cyber. Explanation: Not allowed.")
    assert api.claude_response_text(message('refusal')) == (
        None, "The model refused the request (no stop_details were given).")


def test_unexpected_stop_reason(api):
    assert api.claude_response_text(message('pause_turn', [block('text', 'x')])) == (
        None, "Unexpected stop reason 'pause_turn'; not using the output.")


def test_no_text(api):
    assert api.claude_response_text(message('end_turn', [block('text', '  \n')])) == (
        None, "The model finished normally but returned no text.")


# anthropic_request with a fake Anthropic client: nothing is sent anywhere

class FakeStream:
    def __init__(self, events, final_message, error=None):
        self.events = events
        self.final_message = final_message
        self.error = error

    def __enter__(self):
        if self.error:
            raise self.error
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final_message


CONNECTION_SETTINGS = {'api': 'anthropic', 'base_url': 'https://fake-resource.example.invalid/anthropic/',
                       'api_key': 'fake-key'}


def fake_anthropic(monkeypatch, stream):
    """Replace anthropic.Anthropic with a fake whose messages.stream returns stream (or, if stream is callable,
    calls it with the request parameters)."""

    class FakeAnthropic:
        def __init__(self, api_key, base_url):
            assert (api_key, base_url) == ('fake-key', 'https://fake-resource.example.invalid/anthropic/')
            self.base_url = base_url
            self.messages = SimpleNamespace(stream=stream if callable(stream) else lambda **params: stream)

    monkeypatch.setattr(anthropic, 'Anthropic', FakeAnthropic)


def final_message(stop_reason, text):
    return SimpleNamespace(model='fake-deployment', stop_reason=stop_reason, content=[block('text', text)],
                           usage=SimpleNamespace(input_tokens=120, output_tokens=45, cache_creation_input_tokens=None,
                                                 cache_read_input_tokens=0))


def text_event(text):
    return SimpleNamespace(type='content_block_delta', delta=SimpleNamespace(type='text_delta', text=text))


CONNECTION = ("Using the Anthropic API at https://fake-resource.example.invalid/anthropic/ (model "
              "fake-deployment, effort low)")
USAGE = "namespace(input_tokens=120, output_tokens=45, cache_creation_input_tokens=None, cache_read_input_tokens=0)"
TOKENS = {'input': 120, 'output': 45, 'cache_write': 0, 'cache_read': 0}


def request(api, verbose=False):
    return api.anthropic_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50,
                                 effort='low', label="Test", verbose=verbose)


def test_successful_call_prints_nothing_by_default(api, monkeypatch, capsys):
    fake_anthropic(monkeypatch, FakeStream([text_event('Hi')], final_message('end_turn', 'Hi')))
    assert request(api) == ('Hi', TOKENS)
    assert capsys.readouterr().out == ""


def test_successful_call_with_verbose_prints_the_details(api, monkeypatch, capsys):
    fake_anthropic(monkeypatch, FakeStream([text_event('Hi')], final_message('end_turn', 'Hi')))
    assert request(api, verbose=True) == ('Hi', TOKENS)
    assert capsys.readouterr().out == (
        f"{CONNECTION}\n"
        "Model: fake-deployment  Stop reason: end_turn  Tokens: 120 input, 45 output\n"
        f"Full usage: {USAGE}\n")


def test_cut_off_call_prints_endpoint_deployment_stop_reason_and_usage_by_default(api, monkeypatch, capsys):
    fake_anthropic(monkeypatch, FakeStream([], final_message('max_tokens', 'partial')))
    assert request(api) == (None, TOKENS)
    assert capsys.readouterr().out == (
        "\n--- Error: no usable output ---\n"
        "The response was cut off at the max_tokens limit and is incomplete. "
        "Not returning partial output. Raise the max tokens setting.\n"
        f"{CONNECTION}\n"
        "Model: fake-deployment  Stop reason: max_tokens\n"
        f"Full usage: {USAGE}\n")


def test_cut_off_call_with_verbose_prints_the_details_once(api, monkeypatch, capsys):
    fake_anthropic(monkeypatch, FakeStream([], final_message('max_tokens', 'partial')))
    assert request(api, verbose=True) == (None, TOKENS)
    out = capsys.readouterr().out
    assert out.count(CONNECTION) == 1
    assert out.count(f"Full usage: {USAGE}") == 1
    assert "The response was cut off at the max_tokens limit" in out


def test_connection_error_prints_endpoint_and_deployment_by_default(api, monkeypatch, capsys):
    error = anthropic.APIConnectionError(request=httpx.Request('POST', 'https://fake-resource.example.invalid/'))
    fake_anthropic(monkeypatch, FakeStream([], None, error=error))
    assert request(api) == (None, None)
    assert capsys.readouterr().out == (
        "\n--- Connection error ---\n"
        "Could not reach https://fake-resource.example.invalid/anthropic/.\n"
        "Details: Connection error.\n"
        f"{CONNECTION}\n"
        "No response was received, so there is no stop reason or token usage.\n")


def test_claude_tokens_captures_cache_writes_and_reads(api):
    usage = SimpleNamespace(input_tokens=50, output_tokens=7, cache_creation_input_tokens=1200,
                            cache_read_input_tokens=3400)
    assert api.claude_tokens(usage) == {'input': 50, 'output': 7, 'cache_write': 1200, 'cache_read': 3400}
    # The SDK's cache fields are Optional; not reported counts as 0
    usage = SimpleNamespace(input_tokens=50, output_tokens=7, cache_creation_input_tokens=None,
                            cache_read_input_tokens=None)
    assert api.claude_tokens(usage) == {'input': 50, 'output': 7, 'cache_write': 0, 'cache_read': 0}


def test_claude_usage_field_names_exist_in_the_installed_sdk():
    fields = set(anthropic.types.Usage.model_fields)
    assert {'input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'} <= fields


def capture_request(monkeypatch, reply_text):
    """Fake client that records the request parameters; returns the dict they are recorded in."""
    sent = {}

    def stream(**params):
        sent.update(params)
        return FakeStream([], final_message('end_turn', reply_text))

    fake_anthropic(monkeypatch, stream)
    return sent


def test_request_parameters(api, monkeypatch):
    sent = capture_request(monkeypatch, 'Hi')
    api.anthropic_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50,
                          effort='low', label="Test")
    assert sent == {'model': 'fake-deployment', 'max_tokens': 50, 'system': 'system',
                    'messages': [{'role': 'user', 'content': 'user'}], 'output_config': {'effort': 'low'}}


def test_no_effort_and_no_schema_sends_no_output_config(api, monkeypatch):
    sent = capture_request(monkeypatch, 'Hi')
    api.anthropic_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50)
    assert 'output_config' not in sent


def test_json_schema_is_sent_as_structured_output_with_effort(api, monkeypatch):
    sent = capture_request(monkeypatch, '{"alt_text": "Hi"}')
    schema = {"type": "object", "properties": {"alt_text": {"type": "string"}}, "required": ["alt_text"],
              "additionalProperties": False}
    text, _ = api.anthropic_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50,
                                    effort='low', label="Test", json_schema=schema)
    assert text == '{"alt_text": "Hi"}'
    assert sent['output_config'] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}


def test_image_and_text_parts_become_anthropic_blocks(api, monkeypatch):
    content = [api.image_part('image/png', 'QUJD'), api.text_part('Describe it.')]
    expected = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'QUJD'}},
                {'type': 'text', 'text': 'Describe it.'}]
    assert api.anthropic_content(content) == expected
    assert api.anthropic_content('plain') == 'plain'
    sent = capture_request(monkeypatch, 'Hi')
    api.anthropic_request(CONNECTION_SETTINGS, "system", content, model='fake-deployment', max_tokens=50)
    assert sent['messages'] == [{'role': 'user', 'content': expected}]


def test_llm_request_sends_anthropic_style_through_anthropic_request(api, monkeypatch):
    sent = capture_request(monkeypatch, 'Hi')
    assert api.llm_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50) == (
        'Hi', TOKENS)
    assert sent['model'] == 'fake-deployment'


def test_parallel_calls_can_turn_off_the_bar_and_collect_their_lines(api, monkeypatch, capsys):
    fake_anthropic(monkeypatch, FakeStream([text_event('partial')], final_message('max_tokens', 'partial')))
    lines = []
    result = api.anthropic_request(CONNECTION_SETTINGS, "system", "user", model='fake-deployment', max_tokens=50,
                                   effort='low', label="Test", show_progress=False, log=lines.append)
    assert result == (None, TOKENS)
    # Nothing reaches the screen: no progress bar, and the error lines go to the collector
    assert capsys.readouterr() == ("", "")
    assert "\n".join(lines) + "\n" == (
        "\n--- Error: no usable output ---\n"
        "The response was cut off at the max_tokens limit and is incomplete. "
        "Not returning partial output. Raise the max tokens setting.\n"
        f"{CONNECTION}\n"
        "Model: fake-deployment  Stop reason: max_tokens\n"
        f"Full usage: {USAGE}\n")
