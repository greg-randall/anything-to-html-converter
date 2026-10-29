"""Test setup: exposes the code under test to the tests as a single `api` fixture.

The test files only use `api`, so when the code moves between files only this conftest changes.
"""
import importlib
import json
import sys
import threading
import types
from pathlib import Path

import dotenv
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
# The token counts the fake LLM calls report for each successful call
# The connection the fake runs use; nothing is ever sent to it
FAKE_CONNECTION = {'api': 'anthropic', 'base_url': 'https://llm.example.invalid/', 'api_key': 'fake-key'}
FAKE_TOKENS = {'format': {'input': 8061, 'output': 7987, 'cache_write': 0, 'cache_read': 0},
               'alt_text': {'input': 1981, 'output': 41, 'cache_write': 0, 'cache_read': 0}}

# The functions and constants the tests use, by the doc_improver module that holds them
API_NAMES = {
    'convert': ['convert_pandoc_markdown_to_standard', 'get_csv_content_from_spreadsheet', 'convert_to_markdown',
                'convert_markdown_to_html', 'media_folder_name',
                'flatten_extracted_media'],
    'md_syntax': ['IMAGE_RE', 'find_headings', 'strip_inline_markup', 'tokenize_text'],
    'lists': ['find_ordered_lists'],
    'images': ['protect_images', 'check_image_placeholders', 'restore_images', 'check_restored_images',
               'image_alt_text', 'image_target_path', 'resolve_image_file', 'image_media_type'],
    'alt_text': ['build_alt_text_content', 'image_context', 'format_alt_text', 'insert_alt_text', 'start_alt_text',
                 'finish_alt_text', 'restore_images_with_alt_text', 'AltTextRun', 'parse_alt_text_reply',
                 'ALT_TEXT_SCHEMA'],
    'links': ['trim_bare_url', 'normalize_url', 'parse_link_target', 'analyze_markdown', 'compare_urls',
              'is_link_text_edit'],
    'compare': ['find_mismatches', 'get_context', 'compare_texts'],
    'prompts': ['load_prompt', 'fill'],
    'formatting': ['format_content_with_llm'],
    'cli': ['main'],
    'report': ['print_url_report', 'print_link_text_edits', 'print_list_numbering', 'visible_len',
               'print_mismatch_report', 'print_added_alt_text_report', 'print_preserved_summary', 'format_tokens'],
    'config': ['API_STYLE_NAMES', 'get_alt_text_settings', 'get_connection', 'resolve_format_settings'],
    'llm': ['API_STYLES', 'llm_request', 'text_part', 'image_part'],
    'llm_anthropic': ['claude_response_text', 'claude_tokens', 'anthropic_request', 'anthropic_content'],
    'llm_openai': ['openai_request', 'openai_tokens', 'openai_response_text', 'openai_content'],
    'pricing': ['make_tokens', 'load_prices', 'estimate_cost', 'format_cost', 'cost_summary',
                'print_cost_breakdown', 'print_cost_total'],
}


def _load_code():
    """Import the doc_improver modules without reading .env.

    Importing doc_improver.config calls load_dotenv; the tests set every setting they need themselves, so
    load_dotenv is replaced with a no-op for the import.
    """
    original_load_dotenv = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        modules = {name: importlib.import_module(f"doc_improver.{name}") for name in API_NAMES}
    finally:
        sys.path.remove(str(PROJECT_DIR))
        dotenv.load_dotenv = original_load_dotenv
    return modules


def _make_api():
    modules = _load_code()
    api = types.SimpleNamespace(**{name: getattr(modules[module], name)
                                   for module, names in API_NAMES.items() for name in names})

    def fake_llms(monkeypatch, format_reply=None, alt_text_reply=None):
        """Replace both LLM calls (formatting and alt text) with fakes; no request leaves the machine.

        format_reply(system_prompt, user_prompt) and alt_text_reply(system_prompt, user_content) return the
        reply text (or None for a failed call). A call without a reply function fails the test.
        Like the real calls, the fakes return (text, tokens): tokens is FAKE_TOKENS[kind] when there is a reply,
        else None (as for a call that got no response).
        Returns the list of calls, each a dict with kind ('format' or 'alt_text'), api, system, user,
        and the keyword arguments the call was made with. The calls can run in parallel (the reply functions
        are called from several threads), so the list is in the order the calls started, which can vary
        between runs; filter it by kind.
        """
        calls = []
        calls_lock = threading.Lock()

        def record(call):
            with calls_lock:
                calls.append(call)

        def answer(kind, reply, system_prompt, user_content):
            if reply is None:
                raise AssertionError(f"unexpected {kind} LLM call")
            text = reply(system_prompt, user_content)
            return text, (dict(FAKE_TOKENS[kind]) if text is not None else None)

        def fake_request_for(kind, reply):
            def fake_request(connection, system_prompt, user_content, *, model, max_tokens, effort=None,
                             label="Model", verbose=False, json_schema=None, show_progress=True, log=print):
                record({'kind': kind, 'api': connection['api'], 'connection': connection, 'system': system_prompt,
                        'user': user_content, 'model': model, 'max_tokens': max_tokens, 'effort': effort,
                        'label': label, 'verbose': verbose, 'json_schema': json_schema,
                        'show_progress': show_progress, 'log': log})
                text, tokens = answer(kind, reply, system_prompt, user_content)
                # With a schema, the real call returns JSON: wrap the reply like the API would, unless the test
                # returns raw JSON itself (a reply starting with "{")
                if json_schema and text is not None and not text.lstrip().startswith('{'):
                    text = json.dumps({'alt_text': text})
                return text, tokens
            return fake_request

        monkeypatch.setattr(modules['formatting'], 'llm_request', fake_request_for('format', format_reply))
        monkeypatch.setattr(modules['alt_text'], 'llm_request', fake_request_for('alt_text', alt_text_reply))
        return calls

    def use_prices(monkeypatch, text, path):
        """Write text to path as the price file and make the code read it instead of the project's prices.yaml."""
        path.write_text(text, encoding='utf-8')
        monkeypatch.setattr(modules['pricing'], 'PRICES_PATH', str(path))

    def generate_alt_text(markdown, media_base_dir, settings, prices, verbose=False):
        """Run the alt text steps of a run on markdown without a formatting call: protect the images, start the
        alt text calls, collect them and restore the images with the generated alt text.
        Returns (new_markdown, generated, warnings, costs)."""
        protected, placeholders, _prefix = modules['images'].protect_images(markdown)
        run = modules['alt_text'].start_alt_text(markdown, placeholders, media_base_dir, FAKE_CONNECTION, settings,
                                                 prices, verbose=verbose)
        result = modules['alt_text'].finish_alt_text(run)
        new_markdown = modules['alt_text'].restore_images_with_alt_text(protected, placeholders,
                                                                        result.alt_by_placeholder)
        return new_markdown, result.generated, result.warnings, result.costs

    api.FAKE_CONNECTION = FAKE_CONNECTION
    api.fake_llms = fake_llms
    api.generate_alt_text = generate_alt_text
    api.use_prices = use_prices
    return api


_API = _make_api()


@pytest.fixture
def api():
    return _API
