"""Offline end-to-end runs of main() on a small hand-written markdown file in a temp folder.

Both LLM calls are replaced with fakes (no network). Pandoc runs only on the tiny input written here.
Every setting main() reads is set or removed with monkeypatch, so .env values never matter.
"""
import base64
import io
import os
import re
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

SETTINGS = {
    'LLM_API': 'anthropic',
    'LLM_API_KEY': 'fake-key',
    'FORMAT_MODEL': 'test-format-deployment',
    'FORMAT_MAX_TOKENS': '1000',
    'FORMAT_EFFORT': 'high',
    'ALT_TEXT_MODEL': 'test-alt-deployment',
    'ALT_TEXT_MAX_TOKENS': '200',
    'ALT_TEXT_EFFORT': 'low',
    'ALT_TEXT_CONTEXT_WORDS': '5',
    'ALT_TEXT_CONCURRENCY': '4',
}
# Removed so the runs never depend on the machine's environment (the model calls are fakes anyway)
UNSET = ['LLM_BASE_URL']

INPUT = """# Club News

Our 100^th^ reunion is coming. Register here at https://example.com/register today.

![](pic.png)

Contact jo@example.edu with questions.
"""
# The price file the runs read instead of the project's prices.yaml. test-openai-deployment is left out on
# purpose (its cost is unknown).
TEST_PRICES = """\
test-format-deployment:
  source: test prices
  input: 2.00
  output: 10.00
  cache_write:
  cache_read:
test-alt-deployment:
  source: test prices
  input: 0.10
  output: 0.50
  threshold_input_tokens: 100000
  above_threshold:
    input: 0.50
    output: 2.50
"""
CONTENT_MARKER = "Here's the content to reformat into Markdown:\n\n"
ALT_TEXT = "People gathered at a reunion"
# Long enough for a slow machine, short enough that a sequential implementation fails quickly instead of hanging
BARRIER_TIMEOUT = 5


def calls_of(calls, kind):
    """The recorded calls of one kind. Calls run in parallel, so the order across kinds can vary."""
    return [c for c in calls if c['kind'] == kind]


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def prompt_content(user_prompt):
    """The document content inside the formatting prompt (the prompt ends with it plus a newline)."""
    assert CONTENT_MARKER in user_prompt and user_prompt.endswith("\n")
    return user_prompt.split(CONTENT_MARKER, 1)[1][:-1]


@pytest.fixture
def workspace(api, monkeypatch, tmp_path, tmp_path_factory):
    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)
    for name in UNSET:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'news.md').write_text(INPUT, encoding='utf-8')
    Image.new('RGB', (4, 4), 'red').save(tmp_path / 'pic.png', format='PNG')
    # Outside the workspace, so the tests' checks of the files in it are unchanged
    api.use_prices(monkeypatch, TEST_PRICES, tmp_path_factory.mktemp('prices') / 'prices.yaml')
    project_files = sorted(os.listdir(PROJECT_DIR))
    yield tmp_path
    assert sorted(os.listdir(PROJECT_DIR)) == project_files, "the test wrote into the project folder"


def run_main(api, monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['anything-to-anything.py', *args])
    api.main()


def assert_in_order(out, expected_in_order):
    position = 0
    for expected in expected_in_order:
        found = out.find(expected, position)
        assert found >= 0, f"missing or out of order: {expected!r}\n--- output ---\n{out}"
        position = found + len(expected)


def added_text_section(image):
    return ("\n===== ADDED TEXT - PLEASE CHECK: 1 alt text description(s) were written by the model and added to "
            "the document =====\n"
            f"\n  Image:   news_improved_media/{image}\n"
            f"  Alt:     {ALT_TEXT}\n"
            "\nReview each description for accuracy before publishing.\n")


def extracted_image_name(workspace):
    """The file name pandoc gave the extracted image, in the output's own media folder."""
    names = os.listdir(workspace / 'news_improved_media')
    assert len(names) == 1
    return names[0]


def test_end_to_end_with_alt_text(api, monkeypatch, capsys, workspace):
    calls = api.fake_llms(monkeypatch, format_reply=lambda system, user: prompt_content(user),
                          alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    out = strip_ansi(capsys.readouterr().out)

    improved_md = (workspace / 'news_improved.md').read_text(encoding='utf-8')
    html = (workspace / 'news_improved.html').read_text(encoding='utf-8')
    assert f"![{ALT_TEXT}](" in improved_md
    assert "Register here at https://example.com/register today." in improved_md
    assert html.startswith("<!DOCTYPE html>")
    assert f'alt="{ALT_TEXT}"' in html
    # The intermediate markdown is removed (no --keep-markdown); the input is never touched
    assert not (workspace / 'news_converted.md').exists()
    assert (workspace / 'news.md').read_text(encoding='utf-8') == INPUT

    assert sorted((c['kind'], c['api'], c['model'], c['max_tokens'], c['effort']) for c in calls) == [
        ('alt_text', 'anthropic', 'test-alt-deployment', 200, 'low'),
        ('format', 'anthropic', 'test-format-deployment', 1000, 'high')]
    # Both jobs use the one connection from .env
    assert all(c['connection'] == {'api': 'anthropic', 'base_url': None, 'api_key': 'fake-key'} for c in calls)
    format_call, = calls_of(calls, 'format')
    assert "IMAGE PLACEHOLDERS" in format_call['system']
    assert "![" not in prompt_content(format_call['user'])

    assert not any(c['verbose'] for c in calls)

    # The default output is this short summary, in full
    assert out == (
        f"Converting {workspace / 'news.md'} to markdown... done (1 image protected)\n"
        # 8,061 x $2.00/1M + 7,987 x $10.00/1M = $0.095992
        "Formatting with test-format-deployment (effort high)... done: 8,061 in / 7,987 out tokens, $0.0960\n"
        # 1,981 x $0.10/1M + 41 x $0.50/1M = $0.0002186
        "Alt text with test-alt-deployment (effort low): 1 image... done: 1,981 in / 41 out tokens, $0.0002\n"
        "Total estimated cost (list price): $0.0962\n"
        f"Saved {workspace / 'news_improved.md'} and {workspace / 'news_improved.html'}\n"
        "\nAll link URLs preserved. All original words preserved.\n"
        + added_text_section(extracted_image_name(workspace)))


def test_end_to_end_verbose_shows_the_details(api, monkeypatch, capsys, workspace):
    def format_reply(system, user):
        # Adds a word, which only --verbose lists
        return prompt_content(user).replace("Contact jo@example.edu", "Please contact jo@example.edu")

    calls = api.fake_llms(monkeypatch, format_reply=format_reply,
                          alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--verbose')
    out = strip_ansi(capsys.readouterr().out)
    assert [c['verbose'] for c in calls] == [True, True]

    image = extracted_image_name(workspace)
    assert_in_order(out, [
        f"Detected document: {workspace / 'news.md'}. Converting to Markdown...",
        f"Successfully converted {workspace / 'news.md'} to {workspace / 'news_converted.md'}",
        "Protected 1 image reference(s) with placeholders.",
        # The alt text calls start before the formatting call
        ("Alt text: 1 of 1 image(s) have no alt text; generating it with test-alt-deployment, up to 4 call(s) "
         "at a time, alongside the formatting call."),
        "Formatting content with test-format-deployment (max tokens 1000)",
        ("Formatting with test-format-deployment (effort high)... done: 8,061 in / 7,987 out tokens, $0.0960\n"
         "  Estimated cost (list price) for test-format-deployment (1 request, source: test prices):\n"
         "    input:       8,061 tokens x $2.00 per 1M = $0.0161\n"
         "    output:      7,987 tokens x $10.00 per 1M = $0.0799\n"
         "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
         "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
         "    total:       $0.0960\n"),
        f"Alt text 1/1: ![](news_improved_media/{image})",
        f"Sending news_improved_media/{image} (image/png, ",
        "# Club News\n\nOur 100^th^ reunion is coming. Register here at https://example.com/register today.",
        f"Generated alt text for news_improved_media/{image}: {ALT_TEXT}",
        ("Alt text with test-alt-deployment (effort low): 1 image... done: 1,981 in / 41 out tokens, $0.0002\n"
         "  Estimated cost (list price) for test-alt-deployment (1 request, source: test prices):\n"
         "    input:       1,981 tokens x $0.10 per 1M = $0.0002\n"
         "    output:      41 tokens x $0.50 per 1M = <$0.0001\n"
         "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
         "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
         "    total:       $0.0002\n"),
        "\nTotal estimated cost (list price): $0.0962\n",
        f"Improved markdown saved to {workspace / 'news_improved.md'}",
        f"HTML conversion complete. Final output saved to {workspace / 'news_improved.html'}",
        "\nAll link URLs preserved.\n",
        "\nAll original words preserved. Content preservation verified!\n",
        ("\nNOTE: Words added in the AI-formatted version (outside expected link text edits and list numbering):\n"
         "  - 'please' appears 1 more times\n"),
        "Found 1 content differences (these should be investigated):",
        added_text_section(image),
    ])
    assert "Saved " not in out


def test_end_to_end_reports_link_text_edits_and_missing_words(api, monkeypatch, capsys, workspace):
    def format_reply(system, user):
        content = prompt_content(user)
        assert "Register here at https://example.com/register today." in content
        assert content.startswith("# Club News\n")
        # The dropped word is more than 5 words away from the link edit, so the two are not merged
        return (content.replace("Register here at https://example.com/register today.",
                                "[Register](https://example.com/register) today.")
                .replace("# Club News", "# News", 1))

    api.fake_llms(monkeypatch, format_reply=format_reply, alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--keep-markdown', '--show-diff')
    out = strip_ansi(capsys.readouterr().out)

    assert (workspace / 'news_converted.md').exists()
    # Default mode: what was preserved on one line, every problem in full, link text edits on one line
    assert_in_order(out, [
        f"Saved {workspace / 'news_improved.md'} and {workspace / 'news_improved.html'}\n",
        "\nAll link URLs preserved.\n",
        ("\nWARNING: Words missing in the AI-formatted version:\n  - 'club' appears 1 fewer times\n"
         "\nThis should not happen! Please review the output carefully.\n"),
        'Link text edits (1, expected): "here at" -> ""\n',
        "\nFound 1 content differences (these should be investigated):\n",
        "Content Difference #1 (delete):",
        "===== UNIFIED LINE DIFF =====",
        "+Our 100^th^ reunion is coming. [Register](https://example.com/register) today.",
        added_text_section(extracted_image_name(workspace)),
    ])
    assert "All original words preserved" not in out


def test_url_changes_and_alt_text_warnings_print_in_default_mode(api, monkeypatch, capsys, workspace):
    def format_reply(system, user):
        return prompt_content(user).replace("https://example.com/register", "https://example.com/other")

    api.fake_llms(monkeypatch, format_reply=format_reply, alt_text_reply=lambda system, content: None)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    out = strip_ansi(capsys.readouterr().out)
    image = extracted_image_name(workspace)
    assert_in_order(out, [
        (f"WARNING: no alt text generated for ![](news_improved_media/{image}): the API call failed or returned "
         "no text (see the error above). Its alt text stays empty.\n"),
        ("Alt text with test-alt-deployment (effort low): 1 image... done, 0 of 1 got alt text "
         "(see the warnings above): 0 in / 0 out tokens (1 call returned no usage)\n"
         "Total estimated cost (list price): $0.0960 (1 call reported no token usage, not counted)\n"),
        "\nAll original words preserved.\n",
        ("\nERROR: 1 URL(s) were removed, added or altered in the AI-formatted version. Review these:\n"
         "  - ALTERED: https://example.com/register\n         to: https://example.com/other\n"),
        ("\nWARNING: 1 image(s) did not get alt text:\n"
         f"  - ![](news_improved_media/{image}): the API call failed or returned no text (see the error above)\n"),
    ])
    assert "All link URLs preserved" not in out
    assert "ADDED TEXT" not in out


def test_end_to_end_openai_style_without_alt_text(api, monkeypatch, capsys, workspace):
    calls = api.fake_llms(monkeypatch, format_reply=lambda system, user: prompt_content(user))
    (workspace / 'out').mkdir()
    monkeypatch.delenv('FORMAT_EFFORT')
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--api', 'openai', '--base-url', 'http://localhost:1/v1',
             '--model', 'test-openai-deployment', '--max-tokens', '900', '--no-alt-text',
             '--output', str(workspace / 'out' / 'result.md'))
    out = strip_ansi(capsys.readouterr().out)
    assert [(c['kind'], c['api'], c['model'], c['max_tokens'], c['effort']) for c in calls] == [
        ('format', 'openai', 'test-openai-deployment', 900, None)]
    assert calls[0]['connection'] == {'api': 'openai', 'base_url': 'http://localhost:1/v1', 'api_key': 'fake-key'}
    # No price for the model: the cost is unknown, never guessed
    assert_in_order(out, [
        ("Formatting with test-openai-deployment (effort API default)... done: 8,061 in / 7,987 out tokens, "
         "cost unknown "
         "(no price for test-openai-deployment in prices.yaml)\n"),
        "Alt text generation is off (--no-alt-text).\n",
        "Total estimated cost (list price): unknown (no price for test-openai-deployment in prices.yaml)\n",
    ])
    assert "ADDED TEXT" not in out
    assert (workspace / 'out' / 'result.md').exists()
    assert (workspace / 'out' / 'result.html').exists()


def test_failed_format_call_writes_nothing(api, monkeypatch, capsys, workspace):
    api.fake_llms(monkeypatch, format_reply=lambda system, user: None)
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--no-alt-text')
    out = strip_ansi(capsys.readouterr().out)
    assert "Formatting with test-format-deployment (effort high)... failed: token usage not reported\n" in out
    assert "Total estimated cost (list price): unknown (1 call reported no token usage, not counted)\n" in out
    assert "--- FATAL ERROR ---" in out
    assert "The formatting call to test-format-deployment failed and no formatted content was returned." in out
    assert not (workspace / 'news_improved.md').exists()
    assert not (workspace / 'news_improved.html').exists()


def test_dropped_image_placeholder_writes_nothing(api, monkeypatch, capsys, workspace):
    api.fake_llms(monkeypatch, format_reply=lambda system, user: "Only text now.\n",
                  alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    out = strip_ansi(capsys.readouterr().out)
    # The alt text call ran alongside the formatting call, so it is in the total
    assert "Total estimated cost (list price): $0.0962\n" in out
    assert "--- FATAL ERROR: image placeholders were not preserved ---" in out
    assert "No output was written." in out
    assert not (workspace / 'news_improved.md').exists()


@pytest.mark.parametrize('unset, message', [
    ('LLM_API', "no API style given: set LLM_API in .env or pass --api (anthropic or openai)"),
    ('LLM_API_KEY', "no API key: set LLM_API_KEY in .env (for a local server that needs no key, any text works)"),
    ('FORMAT_MODEL', "no model given: set FORMAT_MODEL in .env or pass --model"),
    ('FORMAT_MAX_TOKENS', "no max tokens given: set FORMAT_MAX_TOKENS in .env or pass --max-tokens"),
])
def test_missing_setting_is_a_usage_error(api, monkeypatch, capsys, workspace, unset, message):
    monkeypatch.delenv(unset)
    calls = api.fake_llms(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        run_main(api, monkeypatch, str(workspace / 'news.md'))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert calls == []


def test_unknown_api_style_in_env_is_a_usage_error(api, monkeypatch, capsys, workspace):
    monkeypatch.setenv('LLM_API', 'azure')
    api.fake_llms(monkeypatch)
    with pytest.raises(SystemExit):
        run_main(api, monkeypatch, str(workspace / 'news.md'))
    assert "LLM_API in .env is 'azure'; expected one of: anthropic, openai" in capsys.readouterr().err


def test_effort_option_works_with_the_openai_style(api, monkeypatch, workspace):
    calls = api.fake_llms(monkeypatch, format_reply=lambda system, user: prompt_content(user))
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--api', 'openai', '--effort', 'medium', '--no-alt-text')
    assert [(c['api'], c['effort']) for c in calls] == [('openai', 'medium')]


def test_invalid_output_extension(api, monkeypatch, capsys, workspace):
    api.fake_llms(monkeypatch)
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--output', str(workspace / 'out.txt'))
    out = capsys.readouterr().out
    assert 'Error: Invalid output format! Got ".txt" but --output must end in .html, .htm or .md' in out
    assert sorted(os.listdir(workspace)) == ['news.md', 'pic.png']


def test_unreadable_input_is_reported(api, monkeypatch, capsys, workspace):
    (workspace / 'notes.xyz').write_text('hello', encoding='utf-8')
    api.fake_llms(monkeypatch)
    run_main(api, monkeypatch, str(workspace / 'notes.xyz'))
    out = capsys.readouterr().out
    assert f"Error: pandoc could not convert {workspace / 'notes.xyz'}:" in out
    assert "Conversion failed. Exiting." in out


# --- Alt text runs in parallel with the formatting call ---

def two_color_input(workspace):
    """Replace the input with two paragraphs, each with its own image (red.png and blue.png)."""
    Image.new('RGB', (4, 4), (255, 0, 0)).save(workspace / 'red.png', format='PNG')
    Image.new('RGB', (4, 4), (0, 0, 255)).save(workspace / 'blue.png', format='PNG')
    (workspace / 'news.md').write_text("# Club News\n\nThe first picture is here.\n\n![](red.png)\n\n"
                                       "The second picture is here.\n\n![](blue.png)\n", encoding='utf-8')


def sent_color(content):
    """The color of the image sent with an alt text request, from its pixels."""
    data = base64.standard_b64decode(content[0]['data'])
    with Image.open(io.BytesIO(data)) as image:
        return {(255, 0, 0): 'red', (0, 0, 255): 'blue'}[image.convert('RGB').getpixel((0, 0))]


def file_color(path):
    with Image.open(path) as image:
        return {(255, 0, 0): 'red', (0, 0, 255): 'blue'}[image.convert('RGB').getpixel((0, 0))]


def test_alt_text_runs_while_formatting_is_in_flight(api, monkeypatch, capsys, workspace):
    # The formatting call and the alt text call each wait until the other has started; run one after the
    # other, the barrier times out and the run fails
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)

    def format_reply(system, user):
        barrier.wait()
        return prompt_content(user)

    def alt_text_reply(system, content):
        barrier.wait()
        return ALT_TEXT

    api.fake_llms(monkeypatch, format_reply=format_reply, alt_text_reply=alt_text_reply)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    assert f"![{ALT_TEXT}](" in (workspace / 'news_improved.md').read_text(encoding='utf-8')


def test_alt_text_context_comes_from_the_original_text(api, monkeypatch, capsys, workspace):
    def format_reply(system, user):
        return prompt_content(user).replace("reunion is coming", "gathering is soon")

    calls = api.fake_llms(monkeypatch, format_reply=format_reply, alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    alt_call, = calls_of(calls, 'alt_text')
    context_text = alt_call['user'][1]['text']
    assert "Our 100^th^ reunion is coming." in context_text
    assert "gathering" not in context_text
    assert "gathering is soon" in (workspace / 'news_improved.md').read_text(encoding='utf-8')


def test_moved_images_keep_their_own_alt_text(api, monkeypatch, capsys, workspace):
    two_color_input(workspace)

    def format_reply(system, user):
        # The model swaps the two paragraphs, images included
        content = prompt_content(user)
        head, first, second = content.split("The ")
        return head + "The " + second.rstrip("\n") + "\n\nThe " + first.rstrip("\n") + "\n"

    api.fake_llms(monkeypatch, format_reply=format_reply,
                  alt_text_reply=lambda system, content: f"A {sent_color(content)} square")
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    improved_md = (workspace / 'news_improved.md').read_text(encoding='utf-8')
    images = re.findall(r'!\[([^\]]*)\]\(([^)]*)\)', improved_md)
    assert improved_md.index("The second picture") < improved_md.index("The first picture")
    assert [alt for alt, _path in images] == ["A blue square", "A red square"]
    for alt, path in images:
        assert alt == f"A {file_color(workspace / path)} square"


def test_failed_format_call_still_totals_the_alt_text_calls(api, monkeypatch, capsys, workspace):
    two_color_input(workspace)
    monkeypatch.setenv('ALT_TEXT_CONCURRENCY', '1')
    alt_started = threading.Event()
    release = threading.Event()

    def format_reply(system, user):
        # Fail only once the first alt text call is running; the second is still waiting for the only thread
        assert alt_started.wait(BARRIER_TIMEOUT)
        # No reply: the formatting call failed

    def alt_text_reply(system, content):
        alt_started.set()
        assert release.wait(BARRIER_TIMEOUT), "the running alt text call was never released"
        return ALT_TEXT

    # Let the running alt text call finish only after the run has cancelled the pending one
    cancel_pending = api.AltTextRun.cancel_pending

    def cancel_then_release(run):
        cancelled = cancel_pending(run)
        release.set()
        return cancelled

    monkeypatch.setattr(api.AltTextRun, 'cancel_pending', cancel_then_release)
    calls = api.fake_llms(monkeypatch, format_reply=format_reply, alt_text_reply=alt_text_reply)
    run_main(api, monkeypatch, str(workspace / 'news.md'))
    out = strip_ansi(capsys.readouterr().out)

    assert len(calls_of(calls, 'alt_text')) == 1
    assert_in_order(out, [
        "Formatting with test-format-deployment (effort high)... failed: token usage not reported\n",
        ("Alt text with test-alt-deployment (effort low): 2 images... stopped (the formatting call failed): "
         "1 call cancelled before sending, 1 of 2 got alt text: 1,981 in / 41 out tokens, $0.0002\n"),
        "--- FATAL ERROR ---",
        # The finished alt text call is in the total; the failed formatting call reported no usage
        "Total estimated cost (list price): $0.0002 (1 call reported no token usage, not counted)\n",
    ])
    assert not (workspace / 'news_improved.md').exists()
    assert not (workspace / 'news_improved.html').exists()
    assert "ADDED TEXT" not in out


@pytest.mark.parametrize('value, message', [
    ('0', "alt text concurrency must be at least 1, got 0"),
    ('many', "ALT_TEXT_CONCURRENCY in .env must be a whole number, got 'many'"),
    (None, ("alt text needs a concurrency limit: set ALT_TEXT_CONCURRENCY in .env, "
            "pass --alt-text-concurrency or pass --no-alt-text")),
])
def test_bad_concurrency_setting_stops_before_any_call(api, monkeypatch, capsys, workspace, value, message):
    if value is None:
        monkeypatch.delenv('ALT_TEXT_CONCURRENCY')
    else:
        monkeypatch.setenv('ALT_TEXT_CONCURRENCY', value)
    calls = api.fake_llms(monkeypatch, format_reply=lambda system, user: prompt_content(user),
                          alt_text_reply=lambda system, content: ALT_TEXT)
    with pytest.raises(SystemExit) as exit_info:
        run_main(api, monkeypatch, str(workspace / 'news.md'))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert calls == []


def test_concurrency_option_overrides_the_setting(api, monkeypatch, capsys, workspace):
    monkeypatch.setenv('ALT_TEXT_CONCURRENCY', '0')
    api.fake_llms(monkeypatch, format_reply=lambda system, user: prompt_content(user),
                  alt_text_reply=lambda system, content: ALT_TEXT)
    run_main(api, monkeypatch, str(workspace / 'news.md'), '--alt-text-concurrency', '2', '--verbose')
    assert "up to 2 call(s) at a time" in strip_ansi(capsys.readouterr().out)
    with pytest.raises(SystemExit):
        run_main(api, monkeypatch, str(workspace / 'news.md'), '--alt-text-concurrency', '0')
    assert "alt text concurrency must be at least 1, got 0" in capsys.readouterr().err
