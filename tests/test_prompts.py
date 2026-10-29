"""Prompts: the text sent to the models is the prompt from config/prompts/*.yaml with its slots filled in.

The expected text is built from the YAML files when the test runs, so editing a prompt's wording never breaks
these tests; a broken slot, a missing rule or text put in the wrong place does.
The content includes braces, backticks and slot-like names to show they are inserted as they are.
"""
import base64
import re
from pathlib import Path

import pytest
import yaml
from PIL import Image

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"
CONTENT = "Hello {world} and `code`, {content} {placeholder_rule} {context} $x \\ end\n\nIMAGEABCD1234N1X\n"
PREFIX = "IMAGEABCD1234N"
CONTEXT = "# Heading\n\nSome {context} text with `backticks` and {braces}."


def prompts(name):
    return yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def render(template, **values):
    """The template with each {slot} replaced by its value. Only slots in the template itself are replaced,
    never slot-like text inside an inserted value."""
    return re.sub(r'\{(\w+)\}', lambda m: values.get(m.group(1), m.group(0)), template)


def format_call(api, monkeypatch, api_style, prefix):
    calls = api.fake_llms(monkeypatch, format_reply=lambda system, user: "formatted")
    connection = {'api': api_style, 'base_url': None, 'api_key': 'fake-key'}
    result = api.format_content_with_llm(CONTENT, connection=connection, model='m', max_tokens=7, effort='high',
                                         image_placeholder_prefix=prefix)
    assert result == ("formatted", {'input': 8061, 'output': 7987, 'cache_write': 0, 'cache_read': 0})
    assert len(calls) == 1
    return calls[0]


def test_format_prompts_with_image_placeholders(api, monkeypatch):
    call = format_call(api, monkeypatch, 'anthropic', PREFIX)
    assert (call['model'], call['max_tokens'], call['effort'], call['label']) == ('m', 7, 'high', 'Formatting')
    format_prompts = prompts("format")
    rule = render(format_prompts['placeholder_rule'], prefix=PREFIX)
    assert PREFIX in rule
    assert call['system'] == render(format_prompts['system'], placeholder_rule=rule)
    assert call['user'] == render(format_prompts['user'], content=CONTENT)


def test_format_prompts_without_image_placeholders(api, monkeypatch):
    call = format_call(api, monkeypatch, 'anthropic', None)
    format_prompts = prompts("format")
    assert call['system'] == render(format_prompts['system'], placeholder_rule="")
    assert call['user'] == render(format_prompts['user'], content=CONTENT)
    assert PREFIX not in call['system']


def test_no_slot_is_left_unfilled(api, monkeypatch):
    for prefix in (PREFIX, None):
        call = format_call(api, monkeypatch, 'anthropic', prefix)
        prompt_part, content_part = call['user'].rsplit(CONTENT, 1)
        assert not re.search(r'\{(placeholder_rule|content|prefix)\}', call['system'] + prompt_part + content_part)


def test_rules_are_in_the_system_prompt_and_the_document_in_the_user_prompt(api, monkeypatch):
    call = format_call(api, monkeypatch, 'anthropic', PREFIX)
    assert "PRESERVE ALL CONTENT" in call['system'] and "PRESERVE ALL CONTENT" not in call['user']
    assert "IMAGE PLACEHOLDERS" in call['system'] and "IMAGE PLACEHOLDERS" not in call['user']
    assert CONTENT not in call['system'] and call['user'].endswith(CONTENT + "\n")


def test_openai_style_gets_the_same_prompts(api, monkeypatch):
    anthropic_call = format_call(api, monkeypatch, 'anthropic', PREFIX)
    openai_call = format_call(api, monkeypatch, 'openai', PREFIX)
    assert (openai_call['api'], openai_call['model'], openai_call['max_tokens'], openai_call['effort']) == (
        'openai', 'm', 7, 'high')
    assert (openai_call['system'], openai_call['user']) == (anthropic_call['system'], anthropic_call['user'])


def test_alt_text_user_prompt(api):
    content = api.build_alt_text_content(b'abc', 'image/png', CONTEXT)
    assert content == [api.image_part('image/png', base64.standard_b64encode(b'abc').decode('utf-8')),
                       api.text_part(render(prompts("alt_text")['user'], context=CONTEXT))]


def test_alt_text_system_prompt_and_context_through_the_whole_path(api, monkeypatch, tmp_path):
    Image.new('RGB', (2, 2)).save(tmp_path / 'a.png', format='PNG')
    calls = api.fake_llms(monkeypatch, alt_text_reply=lambda system, content: "A logo")
    settings = {'model': 'm', 'max_tokens': 10, 'effort': None, 'context_words': 1, 'concurrency': 1}
    api.generate_alt_text("Some text here.\n\n![](a.png)\n", str(tmp_path), settings, {})
    assert len(calls) == 1
    alt_text_prompts = prompts("alt_text")
    assert calls[0]['system'] == alt_text_prompts['system']
    image, text = calls[0]['user']
    assert image['type'] == 'image' and image['media_type'] == 'image/png'
    before, after = alt_text_prompts['user'].split("{context}")
    assert text['text'].startswith(before) and text['text'].endswith(after)
    assert "Some text here." in text['text']


def test_prompt_loader_errors_name_the_file_and_key(api):
    with pytest.raises(KeyError, match=r"'missing' is missing from .*format\.yaml"):
        api.load_prompt('format', 'missing')
    with pytest.raises(FileNotFoundError, match=r"no_such_prompts\.yaml is missing"):
        api.load_prompt('no_such_prompts', 'system')
    with pytest.raises(KeyError, match=r"'system' in .*format\.yaml has no \{content\} slot"):
        api.load_prompt('format', 'system', slots=('content',))


def test_fill_replaces_slots_in_order(api):
    assert api.fill('a {x} {y} {z}', x='{y}', y='Y') == 'a Y Y {z}'
    assert api.fill('{a}', a='{b} $1 \\1') == '{b} $1 \\1'
