"""Numbered lists: finding their item numbers, and telling numbering added by the formatter apart from words."""

# What pandoc made of a small Word document, for the parts with lists:
# Word's own numbering becomes list items and carries on across lists (4., 5.), a sub-list is split off with
# <!-- -->, and numbers typed by hand are escaped (1\.) so they are plain text.
PANDOC_LISTS = """On the Day

1.  Arrive five minutes early.

2.  Check in at the front desk.

3.  Leave the room as you found it.

Cancelling

1\\. Open your confirmation email.

2\\. Select the cancel link.

Setting Up the Projector

4.  Turn on the projector.

<!-- -->

1.  Press the power button.

2.  Wait for the light to turn green.

<!-- -->

5.  Connect your laptop.

Rooms 1, 2 and 3 are on the first floor.
"""

STEPS = "To book a room:\n\nOpen the form.\n\nChoose a date.\n\nEnter the number of people.\n\nSubmit the form.\n"
NUMBERED_STEPS = ("To book a room:\n\n1. Open the form.\n2. Choose a date.\n3. Enter the number of people.\n"
                  "4. Submit the form.\n")


def numbers_of(api, text):
    return [(found['numbers'], found['consecutive']) for found in api.find_ordered_lists(text)]


def test_lists_in_pandoc_output(api):
    assert numbers_of(api, PANDOC_LISTS) == [([1, 2, 3], True), ([4], True), ([1, 2], True), ([5], True)]


def test_item_number_offsets_point_at_the_number(api):
    text = "Intro\n\n1. One\n  12) Twelve\n"
    found = api.find_ordered_lists(text)
    assert [text[o:o + 2] for f in found for o in f['offsets']] == ['1.', '12']


def test_nested_list_is_its_own_list_and_the_outer_list_carries_on(api):
    text = "1. Turn it on.\n   1. Press power.\n   2. Wait.\n2. Connect.\n"
    assert numbers_of(api, text) == [([1, 2], True), ([1, 2], True)]


def test_numbers_out_of_order(api):
    assert numbers_of(api, "1. a\n2. b\n5. c\n") == [([1, 2, 5], False)]
    assert numbers_of(api, "3. a\n4. b\n") == [([3, 4], True)]


def test_blank_lines_and_indented_text_between_items_keep_one_list(api):
    text = "1. a\n\n   more about a\n\n2. b\nlazy line of b\n3. c\n"
    assert numbers_of(api, text) == [([1, 2, 3], True)]


def test_what_ends_a_list(api):
    assert numbers_of(api, "1. a\n\nA paragraph.\n\n2. b\n") == [([1], True), ([2], True)]
    assert numbers_of(api, "1. a\n- bullet\n2. b\n") == [([1], True), ([2], True)]
    assert numbers_of(api, "1. a\n## Heading\n2. b\n") == [([1], True), ([2], True)]
    assert numbers_of(api, "1. a\n2) b\n") == [([1], True), ([2], True)]


def test_not_list_items(api):
    assert numbers_of(api, "1\\. escaped\n2026.no space\nRooms 1. and 2.\n```\n1. in code\n```\n") == []


def test_analyze_markdown_marks_item_numbers(api):
    result = api.analyze_markdown("1. Open [the form](https://example.com/f).\n2. Sign it.\n")
    assert result['tokens'] == ['1', 'Open', 'the', 'form', '2', 'Sign', 'it']
    assert result['list_item'] == [0, None, None, None, 0, None, None]
    assert result['in_label'] == [False, False, True, True, False, False, False]
    assert result['ordered_lists'][0]['positions'] == [0, 4]


def test_plain_steps_becoming_a_numbered_list_is_list_numbering(api):
    result = api.compare_texts(STEPS, NUMBERED_STEPS)
    assert result['added_words'] == {}
    assert result['missing_words'] == {}
    assert result['mismatches'] and all(m['list_numbering'] for m in result['mismatches'])
    assert result['numbering'] == {
        'added': [{'numbers': [1, 2, 3, 4], 'changed': 4, 'start_words': 'Open the form'}],
        'removed': [], 'out_of_order': []}


def test_numbers_out_of_order_stay_added_words_with_a_warning(api):
    improved = NUMBERED_STEPS.replace("3. Enter", "5. Enter").replace("4. Submit", "6. Submit")
    result = api.compare_texts(STEPS, improved)
    assert result['added_words'] == {'1': 1, '2': 1, '5': 1, '6': 1}
    assert result['numbering']['added'] == []
    assert result['numbering']['out_of_order'] == [
        {'numbers': [1, 2, 5, 6], 'changed': 4, 'start_words': 'Open the form'}]


def test_renumbering_a_list_that_carried_on_from_word(api):
    result = api.compare_texts("Setup\n\n4.  Turn on the projector.\n", "Setup\n\n1. Turn on the projector.\n")
    assert result['added_words'] == {}
    assert result['missing_words'] == {}
    assert [m['list_numbering'] for m in result['mismatches']] == [True]
    assert result['numbering']['added'] == [{'numbers': [1], 'changed': 1, 'start_words': 'Turn on the projector'}]
    assert result['numbering']['removed'] == [{'numbers': [4], 'changed': 1, 'start_words': 'Turn on the projector'}]


def test_numbered_list_becoming_bullets_is_numbering_removed(api):
    result = api.compare_texts("1.  Pen\n\n2.  Paper\n", "- Pen\n- Paper\n")
    assert result['missing_words'] == {}
    assert result['numbering']['removed'] == [{'numbers': [1, 2], 'changed': 2, 'start_words': 'Pen'}]


def test_escaped_numbers_becoming_a_list_change_no_words(api):
    result = api.compare_texts("1\\. Open the email.\n\n2\\. Select the link.\n",
                               "1. Open the email.\n2. Select the link.\n")
    assert result['mismatches'] == []
    assert result['numbering'] == {'added': [], 'removed': [], 'out_of_order': []}


def test_a_word_added_next_to_a_new_number_is_still_reported(api):
    result = api.compare_texts("Open the form.\n\nSign it.\n", "1. Now open the form.\n2. Sign it.\n")
    assert result['added_words'] == {'now': 1}
    assert [m['list_numbering'] for m in result['mismatches']] == [False]


def test_numbers_added_in_a_sentence_are_added_words(api):
    result = api.compare_texts("Rooms 1 and 3 are upstairs.\n", "Rooms 1, 2 and 3 are upstairs.\n")
    assert result['added_words'] == {'2': 1}
    assert result['numbering'] == {'added': [], 'removed': [], 'out_of_order': []}


def test_report_lines(api, capsys):
    result = api.compare_texts(STEPS, NUMBERED_STEPS)
    api.print_list_numbering(result)
    assert api.visible_len(capsys.readouterr().out) == len(
        'List numbering (1, expected): numbers added to 4 of 4 items (1-4) at "Open the form"\n')
    api.print_list_numbering(result, verbose=True)
    out = capsys.readouterr().out
    assert "List numbering (1, expected when lines become a numbered list or a numbered list is renumbered):" in out
    assert '  - Numbers added to 4 of 4 items (1-4) at "Open the form"\n' in out


def test_report_warns_about_numbers_out_of_order(api, capsys):
    result = api.compare_texts("Pen\n\nPaper\n", "1. Pen\n3. Paper\n")
    api.print_list_numbering(result)
    out = capsys.readouterr().out
    assert ('WARNING: a numbered list in the AI-formatted version has its numbers out of order (1, 3) '
            'at "Pen". Its new numbers are counted as added words.') in out


def test_nothing_printed_without_numbering_changes(api, capsys):
    api.print_list_numbering(api.compare_texts("Same.\n", "Same.\n"))
    assert capsys.readouterr().out == ""
