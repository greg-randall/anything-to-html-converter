"""Alt text: the context sent with each image, formatting and inserting the reply, and the parallel alt text calls."""
import base64
import io
import re
import threading

import pytest
from PIL import Image

SECTIONS = """# Section One

First paragraph has five words.

Second paragraph here with six words.

![](media/a.png)

After paragraph one two three.

## Section Two

Later text in section two.
"""


def context_for(api, text, image_ref, min_words):
    start = text.index(image_ref)
    return api.image_context(text, start, start + len(image_ref), min_words)


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def test_find_headings_skips_code_fences(api):
    text = "# One\n\n```\n# not a heading\n```\n\n## Two\n"
    assert api.find_headings(text) == [(0, 1, '# One'), (len("# One\n\n```\n# not a heading\n```\n\n"), 2, '## Two')]


def test_context_stops_once_min_words_are_reached(api):
    context, heading = context_for(api, SECTIONS, '![](media/a.png)', 3)
    assert heading == '# Section One'
    assert context == ("# Section One\n\nSecond paragraph here with six words.\n\n"
                       "After paragraph one two three.")


def test_context_crosses_headings_to_reach_min_words(api):
    context, heading = context_for(api, SECTIONS, '![](media/a.png)', 10)
    assert heading == '# Section One'
    assert context == ("# Section One\n\nFirst paragraph has five words.\n\n"
                       "Second paragraph here with six words.\n\nAfter paragraph one two three.\n\n"
                       "## Section Two\n\nLater text in section two.")


def test_context_heading_reached_going_back_is_not_repeated(api):
    context, heading = context_for(api, SECTIONS, '![](media/a.png)', 20)
    assert heading == '# Section One'
    assert context.count('# Section One') == 1
    assert context.startswith('# Section One\n\nFirst paragraph')


def test_context_with_zero_min_words_is_the_heading(api):
    assert context_for(api, SECTIONS, '![](media/a.png)', 0) == ('# Section One', '# Section One')


def test_inline_image_is_shown_as_this_image(api):
    text = "Intro text ![](media/a.png) more text.\n\nNext paragraph here.\n"
    assert context_for(api, text, '![](media/a.png)', 3) == ("Intro text [this image] more text.", None)


def test_other_images_and_image_only_blocks(api):
    text = ("Para one words.\n\n![](media/x.png)\n\nSee ![](media/y.png) here.\n\n![](media/z.png)\n\n"
            "End para.\n")
    assert context_for(api, text, '![](media/z.png)', 1) == ("See [image] here.\n\nEnd para.", None)


def test_format_alt_text(api):
    assert api.format_alt_text('A  photo\nof [the] team \\ x') == 'A photo of \\[the\\] team \\\\ x'


def test_insert_alt_text(api):
    assert api.insert_alt_text('![](media/a.png){width="1in"}', 'Team photo') == (
        '![Team photo](media/a.png){width="1in"}')
    assert api.insert_alt_text('![old [x]](a.png)', 'new') == '![new](a.png)'


def test_build_alt_text_content(api):
    content = api.build_alt_text_content(b'abc', 'image/png', 'Some {braces} context')
    assert content[0] == {'type': 'image', 'media_type': 'image/png',
                          'data': base64.standard_b64encode(b'abc').decode('utf-8')}
    assert content[1]['type'] == 'text'
    assert 'Some {braces} context' in content[1]['text']
    assert len(content) == 2


ADD_ALT_TEXT_MARKDOWN = ("# Head\n\nSome words here.\n\n![](media/a.png)\n\n![Kept alt](media/a.png)\n\n"
                         "![](missing.png)\n\n![](https://example.com/r.png)\n")
ADD_ALT_TEXT_WARNINGS = [
    ("WARNING: no alt text generated for ![](missing.png): file not found (looked for missing.png relative to {}). "
     "Its alt text stays empty."),
    ("WARNING: no alt text generated for ![](https://example.com/r.png): it is not a local file "
     "(https://example.com/r.png); images are not downloaded. Its alt text stays empty."),
]


ALT_PRICES = {'alt-model': {'source': 'test', 'input': 0.10, 'output': 0.50, 'cache_write': None,
                            'cache_read': None}}


def run_add_alt_text(api, monkeypatch, tmp_path, verbose):
    (tmp_path / 'media').mkdir()
    Image.new('RGB', (2, 2)).save(tmp_path / 'media' / 'a.png', format='PNG')
    calls = api.fake_llms(monkeypatch, alt_text_reply=lambda system, content: "Team [photo]\n at  event")
    settings = {'model': 'alt-model', 'max_tokens': 100, 'effort': 'low', 'context_words': 2, 'concurrency': 2}
    return calls, api.generate_alt_text(ADD_ALT_TEXT_MARKDOWN, str(tmp_path), settings, ALT_PRICES, verbose=verbose)


def test_add_alt_text(api, monkeypatch, tmp_path, capsys):
    markdown = ADD_ALT_TEXT_MARKDOWN
    calls, (new_markdown, generated, warnings, costs) = run_add_alt_text(api, monkeypatch, tmp_path, verbose=False)

    assert new_markdown == markdown.replace("![](media/a.png)", "![Team \\[photo\\] at event](media/a.png)", 1)
    assert generated == [('media/a.png', 'Team \\[photo\\] at event', '# Head')]
    assert warnings == [
        ('![](missing.png)', f"file not found (looked for missing.png relative to {tmp_path})"),
        ('![](https://example.com/r.png)',
         "it is not a local file (https://example.com/r.png); images are not downloaded"),
    ]
    assert [(c['kind'], c['model'], c['max_tokens'], c['effort'], c['label'], c['verbose']) for c in calls] == [
        ('alt_text', 'alt-model', 100, 'low', 'Alt text 1/3', False)]
    # One priced request: 1,981 x $0.10/1M + 41 x $0.50/1M
    assert len(costs) == 1
    assert costs[0].cost == 1981 * 0.10 / 1e6 + 41 * 0.50 / 1e6
    assert calls[0]['user'][0]['media_type'] == 'image/png'
    # Default output: the warnings in full, then one summary line; no per-image details
    out = strip_ansi(capsys.readouterr().out)
    assert out == "\n".join([
        ADD_ALT_TEXT_WARNINGS[0].format(tmp_path),
        ADD_ALT_TEXT_WARNINGS[1],
        ("Alt text with alt-model (effort low): 3 images... done, 1 of 3 got alt text (see the warnings above): "
         "1,981 in / 41 out tokens, $0.0002"),
    ]) + "\n"


def test_add_alt_text_verbose(api, monkeypatch, tmp_path, capsys):
    calls, (_new_markdown, generated, _warnings, _costs) = run_add_alt_text(api, monkeypatch, tmp_path,
                                                                            verbose=True)
    assert generated == [('media/a.png', 'Team \\[photo\\] at event', '# Head')]
    assert calls[0]['verbose'] is True
    out = strip_ansi(capsys.readouterr().out)
    expected_in_order = [
        "Alt text: 3 of 4 image(s) have no alt text; generating it with alt-model, up to 2 call(s) at a time",
        "Alt text 1/3: ![](media/a.png)",
        "Sending media/a.png (image/png, ",
        "# Head\n\nSome words here.",
        "Generated alt text for media/a.png: Team \\[photo\\] at event",
        "Alt text 2/3: ![](missing.png)",
        ADD_ALT_TEXT_WARNINGS[0].format(tmp_path),
        "Alt text 3/3: ![](https://example.com/r.png)",
        ADD_ALT_TEXT_WARNINGS[1],
        ("Alt text with alt-model (effort low): 3 images... done, 1 of 3 got alt text (see the warnings above): "
         "1,981 in / 41 out tokens, $0.0002\n"
         "  Estimated cost (list price) for alt-model (1 request, source: test):\n"
         "    input:       1,981 tokens x $0.10 per 1M = $0.0002\n"
         "    output:      41 tokens x $0.50 per 1M = <$0.0001\n"
         "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
         "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
         "    total:       $0.0002\n"),
    ]
    position = 0
    for expected in expected_in_order:
        found = out.find(expected, position)
        assert found >= 0, f"missing or out of order: {expected!r}\n--- output ---\n{out}"
        position = found + len(expected)


def test_add_alt_text_failed_call_leaves_alt_empty(api, monkeypatch, tmp_path, capsys):
    Image.new('RGB', (2, 2)).save(tmp_path / 'a.png', format='PNG')
    markdown = "Text here.\n\n![](a.png)\n"
    api.fake_llms(monkeypatch, alt_text_reply=lambda system, content: None)
    settings = {'model': 'm', 'max_tokens': 10, 'effort': None, 'context_words': 1, 'concurrency': 1}
    new_markdown, generated, warnings, costs = api.generate_alt_text(markdown, str(tmp_path), settings, {})
    assert new_markdown == markdown
    assert generated == []
    assert warnings == [('![](a.png)', "the API call failed or returned no text (see the error above)")]
    # The call returned no usage, so it is not priced (the summary line already says so)
    assert [(c.model, c.tokens, c.cost) for c in costs] == [('m', None, None)]
    out = strip_ansi(capsys.readouterr().out)
    assert out == (
        "WARNING: no alt text generated for ![](a.png): the API call failed or returned no text (see the error "
        "above). Its alt text stays empty.\n"
        "Alt text with m (effort API default): 1 image... done, 0 of 1 got alt text (see the warnings above): "
        "0 in / 0 out tokens (1 call returned no usage)\n")


def test_parse_alt_text_reply(api):
    assert api.parse_alt_text_reply('{"alt_text": "Two people holding signs"}') == ("Two people holding signs", None)
    text, problem = api.parse_alt_text_reply('Two people. Wait, I should not name him.')
    assert text is None and "not valid JSON" in problem and "Wait, I should not name him." in problem
    text, problem = api.parse_alt_text_reply('{"description": "x"}')
    assert text is None and "no alt_text string" in problem
    text, problem = api.parse_alt_text_reply('["x"]')
    assert text is None and "no alt_text string" in problem


def test_add_alt_text_asks_for_structured_output(api, monkeypatch, tmp_path):
    calls, _ = run_add_alt_text(api, monkeypatch, tmp_path, verbose=False)
    assert calls[0]['json_schema'] == api.ALT_TEXT_SCHEMA
    assert api.ALT_TEXT_SCHEMA['required'] == ['alt_text']
    assert api.ALT_TEXT_SCHEMA['additionalProperties'] is False


def test_add_alt_text_warns_and_keeps_alt_empty_on_a_bad_reply(api, monkeypatch, tmp_path, capsys):
    (tmp_path / 'media').mkdir()
    Image.new('RGB', (2, 2)).save(tmp_path / 'media' / 'a.png', format='PNG')
    api.fake_llms(monkeypatch, alt_text_reply=lambda system, content: '{"alt_text": 5}')
    settings = {'model': 'alt-model', 'max_tokens': 100, 'effort': 'low', 'context_words': 2, 'concurrency': 2}
    new_markdown, generated, warnings, _ = api.generate_alt_text(ADD_ALT_TEXT_MARKDOWN, str(tmp_path), settings,
                                                                 ALT_PRICES)
    assert generated == []
    assert "![](media/a.png)" in new_markdown
    assert dict(warnings)["![](media/a.png)"].startswith("the reply had no alt_text string")
    assert "no alt_text string" in capsys.readouterr().out


# --- The calls run in parallel ---

# Long enough for a slow machine, short enough that a sequential implementation fails quickly instead of hanging
BARRIER_TIMEOUT = 5


def color_images(tmp_path, colors):
    """Save one 2x2 PNG per color as <color>.png and return markdown with one paragraph and image per color."""
    for color in colors:
        Image.new('RGB', (2, 2), color).save(tmp_path / f'{color}.png', format='PNG')
    return "".join(f"Text about the {color} one.\n\n![]({color}.png)\n\n" for color in colors)


def image_color(content):
    """The color name of the 2x2 image sent in an alt text request (from its pixels)."""
    data = base64.standard_b64decode(content[0]['data'])
    with Image.open(io.BytesIO(data)) as image:
        return {(255, 0, 0): 'red', (0, 0, 255): 'blue', (0, 128, 0): 'green',
                (255, 255, 0): 'yellow'}[image.convert('RGB').getpixel((0, 0))]


def settings_with(concurrency):
    return {'model': 'alt-model', 'max_tokens': 100, 'effort': 'low', 'context_words': 2,
            'concurrency': concurrency}


def test_alt_text_calls_overlap(api, monkeypatch, tmp_path):
    # Each call waits until all three are in flight at once; one at a time, the barrier times out
    barrier = threading.Barrier(3, timeout=BARRIER_TIMEOUT)

    def reply(system, content):
        barrier.wait()
        return f"A {image_color(content)} square"

    calls = api.fake_llms(monkeypatch, alt_text_reply=reply)
    markdown = color_images(tmp_path, ['red', 'blue', 'green'])
    new_markdown, generated, warnings, costs = api.generate_alt_text(markdown, str(tmp_path), settings_with(3),
                                                                     ALT_PRICES)
    assert warnings == []
    assert len(costs) == 3
    assert [g[:2] for g in generated] == [('red.png', 'A red square'), ('blue.png', 'A blue square'),
                                          ('green.png', 'A green square')]
    assert "![A blue square](blue.png)" in new_markdown
    # No per-call progress bar for parallel calls, and their lines are collected, not printed directly
    assert all(c['show_progress'] is False and c['log'] is not print for c in calls)


def test_concurrency_limit_is_respected(api, monkeypatch, tmp_path):
    barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
    lock = threading.Lock()
    active = {'now': 0, 'max': 0}

    def reply(system, content):
        with lock:
            active['now'] += 1
            active['max'] = max(active['max'], active['now'])
        barrier.wait()
        with lock:
            active['now'] -= 1
        return "A square"

    api.fake_llms(monkeypatch, alt_text_reply=reply)
    markdown = color_images(tmp_path, ['red', 'blue', 'green', 'yellow'])
    _, generated, _, _ = api.generate_alt_text(markdown, str(tmp_path), settings_with(2), ALT_PRICES)
    assert len(generated) == 4
    assert active['max'] == 2


def test_reordered_images_keep_their_own_alt_text(api, monkeypatch, tmp_path):
    api.fake_llms(monkeypatch, alt_text_reply=lambda system, content: f"A {image_color(content)} square")
    markdown = color_images(tmp_path, ['red', 'blue', 'green'])
    protected, placeholders, _ = api.protect_images(markdown)
    run = api.start_alt_text(markdown, placeholders, str(tmp_path), api.FAKE_CONNECTION, settings_with(3),
                             ALT_PRICES)
    result = api.finish_alt_text(run)
    # The "model" puts the images in reverse order
    first, _second, third = placeholders
    reordered = protected.replace(first, 'TMP').replace(third, first).replace('TMP', third)
    restored = api.restore_images_with_alt_text(reordered, placeholders, result.alt_by_placeholder)
    assert re.findall(r'!\[[^\]]*\]\([^)]*\)', restored) == [
        '![A green square](green.png)', '![A blue square](blue.png)', '![A red square](red.png)']


def test_placeholders_must_match_the_text(api, monkeypatch, tmp_path):
    api.fake_llms(monkeypatch)
    markdown = color_images(tmp_path, ['red', 'blue'])
    _, placeholders, _ = api.protect_images(markdown)
    with pytest.raises(ValueError, match="can't be paired with its image"):
        api.start_alt_text(markdown.replace('![](blue.png)', ''), placeholders, str(tmp_path), api.FAKE_CONNECTION,
                           settings_with(1), ALT_PRICES)


def test_cancelled_calls_are_counted_and_finished_calls_priced(api, monkeypatch, tmp_path, capsys):
    started = threading.Event()
    release = threading.Event()

    def reply(system, content):
        started.set()
        assert release.wait(BARRIER_TIMEOUT), "the test never released the first call"
        return "A square"

    calls = api.fake_llms(monkeypatch, alt_text_reply=reply)
    markdown = color_images(tmp_path, ['red', 'blue'])
    _, placeholders, _ = api.protect_images(markdown)
    run = api.start_alt_text(markdown, placeholders, str(tmp_path), api.FAKE_CONNECTION, settings_with(1),
                             ALT_PRICES)
    assert started.wait(BARRIER_TIMEOUT)
    # The first call is running; the second waits for the only thread and is cancelled
    assert run.cancel_pending() == 1
    release.set()
    result = api.finish_alt_text(run, stop_reason="the formatting call failed")
    assert len(calls) == 1
    assert len(result.costs) == 1 and result.costs[0].cost == 1981 * 0.10 / 1e6 + 41 * 0.50 / 1e6
    assert strip_ansi(capsys.readouterr().out) == (
        "Alt text with alt-model (effort low): 2 images... stopped (the formatting call failed): 1 call cancelled "
        "before sending, 1 of 2 got alt text: 1,981 in / 41 out tokens, $0.0002\n")


def test_lines_print_in_document_order_after_the_calls(api, monkeypatch, tmp_path, capsys):
    # The second image's call finishes first; the output still follows the document
    first_done = threading.Event()

    def reply(system, content):
        color = image_color(content)
        if color == 'red':
            assert first_done.wait(BARRIER_TIMEOUT)
            return None
        first_done.set()
        return "A blue square"

    api.fake_llms(monkeypatch, alt_text_reply=reply)
    markdown = color_images(tmp_path, ['red', 'blue'])
    api.generate_alt_text(markdown, str(tmp_path), settings_with(2), ALT_PRICES, verbose=True)
    out = strip_ansi(capsys.readouterr().out)
    order = [out.index("Alt text 1/2: ![](red.png)"),
             out.index("WARNING: no alt text generated for ![](red.png)"),
             out.index("Alt text 2/2: ![](blue.png)"),
             out.index("Generated alt text for blue.png: A blue square"),
             out.index("Alt text with alt-model")]
    assert order == sorted(order)
