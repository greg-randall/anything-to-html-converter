"""The comparison tokenizer: inline markup is stripped by syntax, so only the words a reader sees remain."""


def test_superscript_and_subscript_markers_are_removed_without_a_space(api):
    assert api.tokenize_text("Our 100^th^ reunion") == ['Our', '100th', 'reunion']
    assert api.tokenize_text("H~2~O") == ['H2O']


def test_strikeout_is_not_subscript(api):
    assert api.strip_inline_markup("~~old~~ new") == "~~old~~ new"
    assert api.tokenize_text("~~old~~ new") == ['old', 'new']


def test_html_tags_and_comments_become_a_space(api):
    assert api.tokenize_text("line<br>break") == ['line', 'break']
    assert api.tokenize_text('<span class="x">bold</span> text') == ['bold', 'text']
    assert api.tokenize_text("a<!-- a note -->b") == ['a', 'b']


def test_autolinks_are_not_html_tags(api):
    assert api.strip_inline_markup("<https://example.com>") == "<https://example.com>"
    assert api.strip_inline_markup("<jo@example.edu>") == "<jo@example.edu>"


def test_words_in_any_script_and_underscores_split_words(api):
    assert api.tokenize_text("café snake_case 2026") == ['café', 'snake', 'case', '2026']


def test_attribute_blocks_are_not_words(api):
    result = api.analyze_markdown('# Title {#intro .big}\n\n'
                                  '[Read more](https://example.com/a){.button target="_blank"}')
    assert result['tokens'] == ['Title', 'Read', 'more']
    assert result['urls'] == [{'url': 'https://example.com/a', 'kind': 'link', 'pos': 1}]


def test_attribute_blocks_after_spans_and_code(api):
    result = api.analyze_markdown('[marked]{.mark} and `code`{=html} end')
    assert result['tokens'] == ['marked', 'and', 'code', 'end']


def test_image_references_are_not_words(api):
    result = api.analyze_markdown('Before ![Logo text](media/image1.png){width="2in"} after')
    assert result['tokens'] == ['Before', 'after']
    assert result['urls'] == []
