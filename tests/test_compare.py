"""Token diff, mismatch labels, conversion clean-up and the reports."""
import re

import pandas as pd
import pytest


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def labels(mismatches):
    return [m['type'] for m in mismatches]


def test_find_mismatches_replace_delete_insert(api):
    assert labels(api.find_mismatches(['a', 'b', 'c'], ['a', 'x', 'c'])) == ['replace']
    assert labels(api.find_mismatches(['a', 'b', 'c'], ['a', 'c'])) == ['delete']
    assert labels(api.find_mismatches(['a', 'c'], ['a', 'b', 'c'])) == ['insert']


def test_find_mismatches_is_case_insensitive(api):
    assert api.find_mismatches(['Hello', 'World'], ['hello', 'WORLD']) == []


def test_find_mismatches_ranges_and_parts(api):
    assert api.find_mismatches(['a', 'b', 'c'], ['a', 'x', 'c']) == [
        {'type': 'replace', 'original_start': 1, 'original_end': 2, 'improved_start': 1, 'improved_end': 2,
         'parts': [(1, 2, 1, 2)]}]


def test_nearby_delete_and_insert_merge_into_a_replace(api):
    result = api.find_mismatches(['a', 'b', 'c', 'd', 'e'], ['a', 'c', 'd', 'X', 'e'])
    assert result == [{'type': 'replace', 'original_start': 1, 'original_end': 4, 'improved_start': 1,
                       'improved_end': 4, 'parts': [(1, 2, 1, 1), (4, 4, 3, 4)]}]


def test_distant_mismatches_stay_separate(api):
    original = ['a'] + [f'w{i}' for i in range(10)] + ['b']
    improved = [f'w{i}' for i in range(10)]
    assert labels(api.find_mismatches(original, improved)) == ['delete', 'delete']


def test_get_context(api):
    tokens = [str(i) for i in range(50)]
    assert api.get_context(tokens, 25, 26, context_window=2) == ('23 24 25 26 27', 23, 28)
    assert api.get_context(tokens, 0, 1)[1:] == (0, 21)


def test_convert_pandoc_markdown_to_standard(api):
    convert = api.convert_pandoc_markdown_to_standard
    assert convert('[text]{.underline}') == '**text**'
    assert convert('[[link](https://example.com)]{.underline}') == '**[link](https://example.com)**'
    assert convert('[1]: https://example.com "Title"') == '[1]: https://example.com "Title"'
    assert convert('[Heading]: rest of text') == 'Heading: rest of text'
    assert convert('**[Bold Heading]**') == '**Bold Heading**'
    assert convert('marked{.mark} text') == 'marked text'


def test_get_csv_content_from_spreadsheet(api, tmp_path, capsys):
    path = tmp_path / 'one.xlsx'
    pd.DataFrame({'Name': ['Ann', 'Bo'], 'Year': [2001, 2002]}).to_excel(path, index=False)
    assert api.get_csv_content_from_spreadsheet(str(path)) == 'Name,Year\nAnn,2001\nBo,2002\n'
    assert capsys.readouterr().out == ""
    assert api.get_csv_content_from_spreadsheet(str(path), verbose=True) == 'Name,Year\nAnn,2001\nBo,2002\n'
    assert f"Successfully extracted data from {path}" in capsys.readouterr().out


def test_spreadsheet_with_several_sheets_is_refused(api, tmp_path):
    path = tmp_path / 'two.xlsx'
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({'a': [1]}).to_excel(writer, sheet_name='First', index=False)
        pd.DataFrame({'b': [2]}).to_excel(writer, sheet_name='Second', index=False)
    with pytest.raises(ValueError, match=r'has 2 sheets \(First, Second\)'):
        api.get_csv_content_from_spreadsheet(str(path))


def test_visible_len_ignores_color_codes(api):
    assert api.visible_len('\x1b[31mred\x1b[0m') == 3


def test_print_url_report(api, capsys):
    api.print_url_report([])
    assert strip_ansi(capsys.readouterr().out) == "\nAll link URLs preserved.\n"
    api.print_url_report([{'type': 'altered', 'original': 'u1', 'improved': 'u2'},
                          {'type': 'removed', 'original': 'u3', 'improved': None},
                          {'type': 'added', 'original': None, 'improved': 'u4'}])
    assert strip_ansi(capsys.readouterr().out) == (
        "\nERROR: 3 URL(s) were removed, added or altered in the AI-formatted version. Review these:\n"
        "  - ALTERED: u1\n         to: u2\n  - REMOVED: u3\n  - ADDED:   u4\n")


def test_print_link_text_edits(api, capsys):
    result = api.compare_texts('Apply here at https://example.com/apply today.',
                               '[Apply](https://example.com/apply) today.')
    api.print_link_text_edits(result)
    assert strip_ansi(capsys.readouterr().out) == 'Link text edits (1, expected): "here at" -> ""\n'
    api.print_link_text_edits(result, verbose=True)
    assert strip_ansi(capsys.readouterr().out) == (
        '\nLink text edits (1, expected when turning URLs into descriptive links):\n  - "here at" -> ""\n')


def test_print_link_text_edits_collapses_identical_edits(api, capsys):
    filler = ' '.join(f"w{i}" for i in range(12))
    original = (f"Read here https://a.example/1 {filler} more here https://a.example/2 {filler} "
                f"see here https://a.example/3 {filler} Apply now at https://a.example/4 end.")
    improved = (f"Read [](https://a.example/1) {filler} more [](https://a.example/2) {filler} "
                f"see [](https://a.example/3) {filler} [Apply](https://a.example/4) end.")
    result = api.compare_texts(original, improved)
    assert result['url_changes'] == [] and not result['missing_words']
    api.print_link_text_edits(result)
    assert strip_ansi(capsys.readouterr().out) == (
        'Link text edits (4, expected): "here" -> "" (x3), "now at" -> ""\n')
    api.print_link_text_edits(result, verbose=True)
    assert strip_ansi(capsys.readouterr().out) == (
        '\nLink text edits (4, expected when turning URLs into descriptive links):\n'
        '  - "here" -> "" (x3)\n  - "now at" -> ""\n')


def test_print_preserved_summary(api, capsys):
    api.print_preserved_summary({'url_changes': [], 'missing_words': {}})
    assert strip_ansi(capsys.readouterr().out) == "\nAll link URLs preserved. All original words preserved.\n"
    api.print_preserved_summary({'url_changes': [{'type': 'removed'}], 'missing_words': {}})
    assert strip_ansi(capsys.readouterr().out) == "\nAll original words preserved.\n"
    api.print_preserved_summary({'url_changes': [{'type': 'removed'}], 'missing_words': {'x': 1}})
    assert capsys.readouterr().out == ""


def test_format_tokens(api):
    assert api.format_tokens(api.make_tokens(input=8061, output=7987)) == "8,061 in / 7,987 out tokens"
    # Cache writes and reads are shown only when there are any
    assert api.format_tokens(api.make_tokens(input=100, output=50, cache_write=300, cache_read=2000)) == (
        "100 in / 50 out / 300 cache write / 2,000 cache read tokens")
    assert api.format_tokens(api.make_tokens(input=100, output=50, cache_read=2000)) == (
        "100 in / 50 out / 2,000 cache read tokens")
    assert api.format_tokens(None) == "token usage not reported"


def test_print_mismatch_report(api, capsys):
    result = api.compare_texts('One two three four.', 'One two four.')
    api.print_mismatch_report(result, detailed=True)
    assert strip_ansi(capsys.readouterr().out) == (
        "\nFound 1 content differences (these should be investigated):\n"
        "\n❌ Content Difference #1 (delete):\n"
        "   ORIGINAL: ...One two [[ three ]] four...\n"
        "   IMPROVED: ...One two [[ [[NONE]] ]] four...\n")


def test_print_mismatch_report_lists_repeats_by_reference(api, capsys):
    segment = ' '.join('abcdefghijklmnopqrstuv')
    result = api.compare_texts(f"{segment} X {segment} X {segment}", f"{segment} {segment} {segment}")
    api.print_mismatch_report(result, detailed=True)
    out = strip_ansi(capsys.readouterr().out)
    assert "Found 2 content differences (these should be investigated):" in out
    assert "\nContent Difference #2 (delete): same text as Content Difference #1 (shown above)\n" in out
    assert "\n1 of 2 differences repeat an earlier one and were listed by reference only.\n" in out


def test_print_mismatch_report_trims_long_lines(api, capsys):
    words = ' '.join(f"word{i:03d}" for i in range(40))
    result = api.compare_texts(f"{words} gone {words}", f"{words} {words}")
    api.print_mismatch_report(result, detailed=True)
    lines = [strip_ansi(line) for line in capsys.readouterr().out.splitlines()
             if 'ORIGINAL:' in line or 'IMPROVED:' in line]
    # At most 10 words are trimmed from each side
    before = ' '.join(f"word{i:03d}" for i in range(30, 40))
    after = ' '.join(f"word{i:03d}" for i in range(10))
    assert lines == [f"   ORIGINAL: ...{before} [[ gone ]] {after}...",
                     f"   IMPROVED: ...{before} [[ [[NONE]] ]] {after}..."]


def test_print_mismatch_report_without_details(api, capsys):
    result = api.compare_texts('One two three.', 'One three.')
    api.print_mismatch_report(result, detailed=False)
    assert capsys.readouterr().out == "Run with --detailed-comparison to see content differences.\n"


def test_print_added_alt_text_report(api, capsys):
    api.print_added_alt_text_report([('media/a.png', 'Team photo', '# Head'), ('media/b.png', 'Logo', None)],
                                    [('![](x.png)', 'file not found')])
    out = strip_ansi(capsys.readouterr().out)
    assert ("===== ADDED TEXT - PLEASE CHECK: 2 alt text description(s) were written by the model and added "
            "to the document =====") in out
    assert "  Image:   media/a.png\n  Alt:     Team photo\n\n  Image:   media/b.png\n  Alt:     Logo\n" in out
    assert "Heading:" not in out
    assert "WARNING: 1 image(s) did not get alt text:\n  - ![](x.png): file not found\n" in out


def test_markdown_the_format_prompt_asks_for_is_not_a_content_change(api):
    """Each structure the format prompt asks for (heading, hard line breaks, block quote, table, bulleted list,
    mailto link, horizontal rule, kept escape) adds no words and changes no URLs."""
    original = ("Annual Report\n\nVisit Us\n\nMain Office\n100 Main Street\nSpringfield\n\n"
                "The guide says: \"Read the rules first.\"\n\nName, Room\nAda, 101\n\n"
                "First open the form, then sign it.\n\nWrite to jo@example.edu with questions.\n\n"
                "2026\\. was a good year.\n\nNext Topic\n\nMore text.\n")
    improved = ("# Annual Report\n\n## Visit Us\n\nMain Office\\\n100 Main Street\\\nSpringfield\n\n"
                "The guide says:\n\n> \"Read the rules first.\"\n\n| Name | Room |\n|---|---|\n| Ada | 101 |\n\n"
                "- First open the form,\n- then sign it.\n\n"
                "Write to [jo@example.edu](mailto:jo@example.edu) with questions.\n\n"
                "2026\\. was a good year.\n\n---\n\n## Next Topic\n\nMore text.\n")
    result = api.compare_texts(original, improved)
    assert result['missing_words'] == {}
    assert result['added_words'] == {}
    assert result['url_changes'] == []
    assert result['mismatches'] == []
