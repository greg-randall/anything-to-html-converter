"""Settings from the command line and the environment (set here with monkeypatch, never from .env), and the
blank.env template."""
import io
import re
from pathlib import Path
from types import SimpleNamespace

import dotenv
import pytest

ALT_TEXT_ENV = ['ALT_TEXT_MODEL', 'ALT_TEXT_MAX_TOKENS', 'ALT_TEXT_EFFORT', 'ALT_TEXT_CONTEXT_WORDS',
                'ALT_TEXT_CONCURRENCY']


class ParserError(Exception):
    pass


class FakeParser:
    """Stands in for argparse's parser: error raises, so a test can check the message."""

    def error(self, message):
        raise ParserError(message)


def args(**overrides):
    values = {'alt_text_model': None, 'model': 'format-model', 'alt_text_max_tokens': None, 'alt_text_effort': None,
              'alt_text_context_words': None, 'alt_text_concurrency': None}
    values.update(overrides)
    return SimpleNamespace(**values)


def clear_env(monkeypatch):
    for name in ALT_TEXT_ENV:
        monkeypatch.delenv(name, raising=False)


def test_alt_text_settings_from_environment(api, monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('ALT_TEXT_MODEL', 'alt-model')
    monkeypatch.setenv('ALT_TEXT_MAX_TOKENS', '4000')
    monkeypatch.setenv('ALT_TEXT_EFFORT', 'low')
    monkeypatch.setenv('ALT_TEXT_CONTEXT_WORDS', '50')
    monkeypatch.setenv('ALT_TEXT_CONCURRENCY', '8')
    assert api.get_alt_text_settings(args()) == {'model': 'alt-model', 'max_tokens': 4000, 'effort': 'low',
                                                 'context_words': 50, 'concurrency': 8}


def test_alt_text_options_override_environment(api, monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('ALT_TEXT_MODEL', 'alt-model')
    monkeypatch.setenv('ALT_TEXT_CONCURRENCY', '8')
    settings = api.get_alt_text_settings(args(alt_text_model='cli-model', alt_text_max_tokens=10,
                                              alt_text_effort='high', alt_text_context_words=0,
                                              alt_text_concurrency=1))
    assert settings == {'model': 'cli-model', 'max_tokens': 10, 'effort': 'high', 'context_words': 0,
                        'concurrency': 1}


def test_alt_text_model_falls_back_to_the_formatting_model(api, monkeypatch):
    clear_env(monkeypatch)
    settings = api.get_alt_text_settings(args(alt_text_max_tokens=10, alt_text_context_words=5,
                                              alt_text_concurrency=2))
    assert settings['model'] == 'format-model'
    assert settings['effort'] is None


def test_alt_text_errors(api, monkeypatch):
    clear_env(monkeypatch)
    assert api.get_alt_text_settings(args()) == (
        "alt text needs a max tokens setting: set ALT_TEXT_MAX_TOKENS in .env, "
        "pass --alt-text-max-tokens or pass --no-alt-text")
    monkeypatch.setenv('ALT_TEXT_MAX_TOKENS', 'lots')
    assert api.get_alt_text_settings(args()) == "ALT_TEXT_MAX_TOKENS in .env must be a whole number, got 'lots'"
    assert api.get_alt_text_settings(args(alt_text_max_tokens=0)) == (
        "alt text max tokens must be a positive number, got 0")
    assert api.get_alt_text_settings(args(alt_text_max_tokens=5)) == (
        "alt text needs a context size: set ALT_TEXT_CONTEXT_WORDS in .env, "
        "pass --alt-text-context-words or pass --no-alt-text")
    monkeypatch.setenv('ALT_TEXT_CONTEXT_WORDS', 'some')
    assert api.get_alt_text_settings(args(alt_text_max_tokens=5)) == (
        "ALT_TEXT_CONTEXT_WORDS in .env must be a whole number, got 'some'")
    assert api.get_alt_text_settings(args(alt_text_max_tokens=5, alt_text_context_words=-1)) == (
        "alt text context words can't be negative, got -1")


def test_alt_text_concurrency_is_validated(api, monkeypatch):
    clear_env(monkeypatch)
    valid = {'alt_text_max_tokens': 5, 'alt_text_context_words': 5}
    assert api.get_alt_text_settings(args(**valid)) == (
        "alt text needs a concurrency limit: set ALT_TEXT_CONCURRENCY in .env, "
        "pass --alt-text-concurrency or pass --no-alt-text")
    for value in ('lots', '2.5'):
        monkeypatch.setenv('ALT_TEXT_CONCURRENCY', value)
        assert api.get_alt_text_settings(args(**valid)) == (
            f"ALT_TEXT_CONCURRENCY in .env must be a whole number, got {value!r}")
    for value in ('0', '-3'):
        monkeypatch.setenv('ALT_TEXT_CONCURRENCY', value)
        assert api.get_alt_text_settings(args(**valid)) == f"alt text concurrency must be at least 1, got {value}"
    monkeypatch.setenv('ALT_TEXT_CONCURRENCY', '3')
    assert api.get_alt_text_settings(args(**valid))['concurrency'] == 3
    # The command line overrides .env and is checked the same way
    assert api.get_alt_text_settings(args(alt_text_concurrency=0, **valid)) == (
        "alt text concurrency must be at least 1, got 0")
    assert api.get_alt_text_settings(args(alt_text_concurrency=5, **valid))['concurrency'] == 5


def test_connection(api, monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'the-key')
    assert api.get_connection(SimpleNamespace(api='openai', base_url='http://localhost:11434/v1'), FakeParser()) == {
        'api': 'openai', 'base_url': 'http://localhost:11434/v1', 'api_key': 'the-key'}
    # A blank base URL means the API's official endpoint
    assert api.get_connection(SimpleNamespace(api='anthropic', base_url=''), FakeParser()) == {
        'api': 'anthropic', 'base_url': None, 'api_key': 'the-key'}


@pytest.mark.parametrize('api_style, key, message', [
    (None, 'k', "no API style given: set LLM_API in .env or pass --api (anthropic or openai)"),
    ('gemini', 'k', "LLM_API in .env is 'gemini'; expected one of: anthropic, openai"),
    ('openai', None, "no API key: set LLM_API_KEY in .env (for a local server that needs no key, any text works)"),
])
def test_connection_errors(api, monkeypatch, api_style, key, message):
    if key is None:
        monkeypatch.delenv('LLM_API_KEY', raising=False)
    else:
        monkeypatch.setenv('LLM_API_KEY', key)
    with pytest.raises(ParserError, match=None) as error:
        api.get_connection(SimpleNamespace(api=api_style, base_url=None), FakeParser())
    assert str(error.value) == message


def test_format_settings_from_environment_and_options(api, monkeypatch):
    monkeypatch.setenv('FORMAT_MODEL', 'env-model')
    monkeypatch.setenv('FORMAT_MAX_TOKENS', '64000')
    monkeypatch.setenv('FORMAT_EFFORT', 'high')
    settings = SimpleNamespace(model=None, max_tokens=None, effort=None)
    api.resolve_format_settings(settings, FakeParser())
    assert vars(settings) == {'model': 'env-model', 'max_tokens': 64000, 'effort': 'high'}
    settings = SimpleNamespace(model='cli-model', max_tokens=5, effort='low')
    api.resolve_format_settings(settings, FakeParser())
    assert vars(settings) == {'model': 'cli-model', 'max_tokens': 5, 'effort': 'low'}
    monkeypatch.delenv('FORMAT_EFFORT')
    settings = SimpleNamespace(model=None, max_tokens=None, effort=None)
    api.resolve_format_settings(settings, FakeParser())
    assert settings.effort is None


@pytest.mark.parametrize('env, options, message', [
    ({'FORMAT_MAX_TOKENS': 'lots'}, {}, "FORMAT_MAX_TOKENS in .env must be a whole number, got 'lots'"),
    ({}, {'max_tokens': 0}, "max tokens must be a positive number, got 0"),
])
def test_format_settings_errors(api, monkeypatch, env, options, message):
    monkeypatch.setenv('FORMAT_MODEL', 'm')
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    settings = SimpleNamespace(**{'model': None, 'max_tokens': None, 'effort': None, **options})
    with pytest.raises(ParserError) as error:
        api.resolve_format_settings(settings, FakeParser())
    assert str(error.value) == message


def test_every_api_style_has_a_request_function(api):
    assert api.API_STYLE_NAMES == ('anthropic', 'openai')
    assert tuple(api.API_STYLES) == api.API_STYLE_NAMES


def test_env_file_uses_the_last_line_when_a_name_is_set_twice(tmp_path):
    # The README tells people they can keep both LLM_API lines and comment one out; if both are left in,
    # the last one wins
    import dotenv
    env_file = tmp_path / '.env'
    env_file.write_text("LLM_API=anthropic\n# LLM_API=ignored\nLLM_API=openai\n", encoding='utf-8')
    assert dotenv.dotenv_values(env_file)['LLM_API'] == 'openai'


BLANK_ENV = Path(__file__).resolve().parent.parent / "blank.env"


def blank_env_settings(text):
    return dotenv.dotenv_values(stream=io.StringIO(text))


def test_blank_env_is_ready_for_anthropic_apart_from_the_key(api):
    values = blank_env_settings(BLANK_ENV.read_text(encoding='utf-8'))
    assert values['LLM_API_KEY'] == ''
    assert values['LLM_BASE_URL'] == ''
    assert {name: values[name] for name in ('LLM_API', 'FORMAT_MODEL', 'FORMAT_EFFORT', 'ALT_TEXT_MODEL',
                                            'ALT_TEXT_EFFORT')} == {
        'LLM_API': 'anthropic', 'FORMAT_MODEL': 'claude-sonnet-5-5', 'FORMAT_EFFORT': 'high',
        'ALT_TEXT_MODEL': 'claude-sonnet-5-5', 'ALT_TEXT_EFFORT': 'low'}
    assert values['LLM_API'] in api.API_STYLE_NAMES
    for name in ('FORMAT_MAX_TOKENS', 'ALT_TEXT_MAX_TOKENS', 'ALT_TEXT_CONTEXT_WORDS', 'ALT_TEXT_CONCURRENCY'):
        assert values[name].isdigit() and int(values[name]) > 0


def test_blank_env_switches_to_openai_when_its_block_is_uncommented(api):
    text = BLANK_ENV.read_text(encoding='utf-8')
    openai_block = text[text.index("# --- OpenAI ---"):text.index("# --- Both ---")]
    uncommented = "\n".join(line[2:] if line.startswith("# ") and "=" in line else line
                            for line in openai_block.splitlines())
    values = blank_env_settings(text.replace(openai_block, uncommented + "\n"))
    assert {name: values[name] for name in ('LLM_API', 'FORMAT_MODEL', 'ALT_TEXT_MODEL')} == {
        'LLM_API': 'openai', 'FORMAT_MODEL': 'gpt-6.1-terra', 'ALT_TEXT_MODEL': 'gpt-6.1-terra'}
    assert values['LLM_API'] in api.API_STYLE_NAMES


def test_models_in_blank_env_have_prices(api):
    text = BLANK_ENV.read_text(encoding='utf-8')
    models = set(re.findall(r'^(?:# )?(?:FORMAT|ALT_TEXT)_MODEL=(\S+)$', text, re.MULTILINE))
    assert models == {'claude-sonnet-5-5', 'gpt-6.1-terra'}
    prices = api.load_prices()
    assert all(prices[model]['input'] is not None and prices[model]['output'] is not None for model in models)
