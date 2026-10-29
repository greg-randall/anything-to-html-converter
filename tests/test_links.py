"""Links and URLs: URLs are kept out of the word stream and compared structurally."""


def urls(*values):
    return [{'url': value} for value in values]


def test_inline_link_label_words_and_target(api):
    result = api.analyze_markdown('See [the Report](https://example.com/report "Title") now.')
    assert result['tokens'] == ['See', 'the', 'Report', 'now']
    assert result['in_label'] == [False, True, True, False]
    assert result['urls'] == [{'url': 'https://example.com/report', 'kind': 'link', 'pos': 1}]


def test_bare_url_keeps_sentence_punctuation_out(api):
    result = api.analyze_markdown('Visit https://example.com/a. Then go.')
    assert result['tokens'] == ['Visit', 'Then', 'go']
    assert result['urls'] == [{'url': 'https://example.com/a', 'kind': 'bare', 'pos': 1}]


def test_email_and_autolink_are_bare_urls(api):
    result = api.analyze_markdown('Email jo@example.edu or <https://example.com/x> today')
    assert result['tokens'] == ['Email', 'or', 'today']
    assert result['urls'] == [{'url': 'mailto:jo@example.edu', 'kind': 'bare', 'pos': 1},
                              {'url': 'https://example.com/x', 'kind': 'bare', 'pos': 2}]


def test_html_link(api):
    result = api.analyze_markdown('Go <a href="https://example.com/p">Click me</a> now')
    assert result['tokens'] == ['Go', 'Click', 'me', 'now']
    assert result['in_label'] == [False, True, True, False]
    assert result['urls'] == [{'url': 'https://example.com/p', 'kind': 'link', 'pos': 1}]


def test_reference_link_url_comes_from_the_definition(api):
    result = api.analyze_markdown('Read [Docs][d] here.\n\n[d]: https://example.com/docs "T"\n')
    assert result['tokens'] == ['Read', 'Docs', 'here']
    assert result['in_label'] == [False, True, False]
    assert result['urls'] == [{'url': 'https://example.com/docs', 'kind': 'link', 'pos': 3}]


def test_url_inside_a_link_label_is_not_recorded(api):
    result = api.analyze_markdown('[https://example.com/shown](https://example.com/target)')
    assert result['urls'] == [{'url': 'https://example.com/target', 'kind': 'link', 'pos': 0}]


def test_trim_bare_url(api):
    assert api.trim_bare_url('https://en.wikipedia.org/wiki/Foo_(bar)).') == 'https://en.wikipedia.org/wiki/Foo_(bar)'
    assert api.trim_bare_url('https://example.com/a,') == 'https://example.com/a'
    assert api.trim_bare_url('https://example.com/a]') == 'https://example.com/a'


def test_normalize_url(api):
    assert api.normalize_url('<https://example.com/a\\_b>') == 'https://example.com/a_b'
    assert api.normalize_url('jo@example.edu') == 'mailto:jo@example.edu'
    assert api.normalize_url('mailto:jo@example.edu') == 'mailto:jo@example.edu'
    assert api.normalize_url('https://example.com/@user') == 'https://example.com/@user'


def test_parse_link_target(api):
    assert api.parse_link_target('https://example.com "Title"') == 'https://example.com'
    assert api.parse_link_target('<https://example.com/a b> "T"') == '<https://example.com/a b>'
    assert api.parse_link_target('  ') == ''


def test_compare_urls_moved_url_is_not_a_change(api):
    assert api.compare_urls(urls('a', 'b'), urls('b', 'a')) == []


def test_compare_urls_altered_removed_added(api):
    assert api.compare_urls(urls('a', 'b', 'c'), urls('a', 'x', 'c')) == [
        {'type': 'altered', 'original': 'b', 'improved': 'x'}]
    assert api.compare_urls(urls('a', 'b'), urls('a')) == [{'type': 'removed', 'original': 'b', 'improved': None}]
    assert api.compare_urls(urls('a'), urls('a', 'c')) == [{'type': 'added', 'original': None, 'improved': 'c'}]


def test_compare_texts_reports_a_changed_link_url(api):
    result = api.compare_texts('[a](https://example.com/1)', '[a](https://example.com/2)')
    assert result['url_changes'] == [
        {'type': 'altered', 'original': 'https://example.com/1', 'improved': 'https://example.com/2'}]


def test_removing_here_at_before_a_url_that_became_a_link_is_a_link_text_edit(api):
    result = api.compare_texts('Apply here at https://example.com/apply today.',
                               '[Apply](https://example.com/apply) today.')
    assert result['url_changes'] == []
    assert result['missing_words'] == {}
    assert result['added_words'] == {}
    assert len(result['mismatches']) == 1
    assert result['mismatches'][0]['link_text_edit'] is True


def test_rewritten_link_label_is_a_link_text_edit(api):
    result = api.compare_texts('[click here](https://example.com/form)',
                               '[complete the form](https://example.com/form)')
    assert [m['link_text_edit'] for m in result['mismatches']] == [True]
    assert result['missing_words'] == {}


def test_dropped_word_outside_a_link_is_not_a_link_text_edit(api):
    result = api.compare_texts('Call us today.', 'Call us.')
    assert result['missing_words'] == {'today': 1}
    assert [(m['type'], m['link_text_edit']) for m in result['mismatches']] == [('delete', False)]


def test_word_dropped_before_a_url_that_stayed_bare_is_not_a_link_text_edit(api):
    result = api.compare_texts('See here https://example.com/a', 'See https://example.com/a')
    assert result['missing_words'] == {'here': 1}
    assert [m['link_text_edit'] for m in result['mismatches']] == [False]


def test_is_link_text_edit_directly(api):
    original = api.analyze_markdown('Apply here at https://example.com/apply')
    improved = api.analyze_markdown('[Apply](https://example.com/apply)')
    mismatch = {'parts': [(1, 3, 1, 1)]}
    assert api.is_link_text_edit(mismatch, original, improved, {3}) is True
    assert api.is_link_text_edit(mismatch, original, improved, set()) is False
