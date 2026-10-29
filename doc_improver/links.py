"""Links and URLs: splitting markdown into the words a reader sees and the URLs it links to, and comparing URLs."""
import difflib
import re
from collections import Counter

from .lists import find_ordered_lists
from .md_syntax import (
    ATTRIBUTE_BLOCK_RE,
    IMAGE_RE,
    LINK_SYNTAX_RE,
    URL_SCHEME_RE,
    tokenize_text,
)


def trim_bare_url(url):
    """Remove trailing punctuation that ends the sentence rather than the URL (and unbalanced closing brackets)."""
    while url:
        last = url[-1]
        sentence_punctuation = last in '.,;:!?\'"*_\\'
        unbalanced_bracket = ((last == ')' and url.count(')') > url.count('('))
                              or (last == ']' and url.count(']') > url.count('[')))
        if sentence_punctuation or unbalanced_bracket:
            url = url[:-1]
        else:
            break
    return url


def normalize_url(raw_url):
    """Normalize a URL for comparison: drop <...> and backslash escapes; a plain email address becomes mailto:."""
    url = raw_url.strip()
    if url.startswith('<') and url.endswith('>'):
        url = url[1:-1]
    url = re.sub(r'\\(.)', r'\1', url)
    if '@' in url and not URL_SCHEME_RE.match(url) and '/' not in url:
        url = 'mailto:' + url
    return url


def parse_link_target(target):
    """Return the URL from an inline link target, dropping an optional title."""
    target = target.strip()
    if target.startswith('<') and '>' in target:
        return target[:target.index('>') + 1]
    return target.split()[0] if target else ''


def analyze_markdown(text):
    """Split markdown into the word tokens a reader sees and the URLs it links to.

    URLs (link targets, autolinks, bare URLs and email addresses) are kept out of the word stream.
    Returns a dict with:
      tokens   - list of word tokens
      in_label - list of bools, True where the token is inside a link label
      list_item - for each token, the index in ordered_lists of the list whose item number it is, else None
      ordered_lists - the numbered lists (lists.find_ordered_lists), each with 'positions': the token
                 position of each item number
      urls     - list of dicts {url, kind, pos}: kind is 'link' (a link target with its own label) or
                 'bare' (a URL shown as itself: bare URL, autolink or email); pos is the number of tokens
                 before it
    """
    # Images are checked separately (they are protected with placeholders), so skip them here
    text = IMAGE_RE.sub(' ', text)

    # Remove pandoc attribute blocks ({#id .class key=value}) after links, spans, headings, etc.
    text = ATTRIBUTE_BLOCK_RE.sub('', text)

    ordered_lists = find_ordered_lists(text)
    for found in ordered_lists:
        found['positions'] = []
    # Offset of each item number in text -> index of its list
    item_numbers = {offset: index for index, found in enumerate(ordered_lists) for offset in found['offsets']}

    result = {'tokens': [], 'in_label': [], 'list_item': [], 'urls': [], 'ordered_lists': ordered_lists}
    _scan_links(text, result, in_label=False, item_numbers=item_numbers)
    return result


def _scan_links(text, result, in_label, item_numbers=None):
    """Add the tokens and URLs of text to result. URLs inside a link label are not recorded (the link's target is).

    item_numbers maps the offset in text of each list item number to its list (see analyze_markdown); each
    number becomes one token marked with its list. It is None for a link label, which can't hold a list item.
    """

    def add_plain_tokens(segment):
        segment_tokens = tokenize_text(segment)
        result['tokens'].extend(segment_tokens)
        result['in_label'].extend([in_label] * len(segment_tokens))
        result['list_item'].extend([None] * len(segment_tokens))

    def add_tokens(segment, start):
        """Add the tokens of segment, which starts at offset start in text."""
        if not item_numbers:
            add_plain_tokens(segment)
            return
        done = 0
        for offset in sorted(o for o in item_numbers if start <= o < start + len(segment)):
            add_plain_tokens(segment[done:offset - start])
            number = re.match(r'\d+', segment[offset - start:]).group()
            index = item_numbers[offset]
            result['ordered_lists'][index]['positions'].append(len(result['tokens']))
            result['tokens'].append(number)
            result['in_label'].append(in_label)
            result['list_item'].append(index)
            done = offset - start + len(number)
        add_plain_tokens(segment[done:])

    def add_url(raw_url, kind):
        if not in_label:
            result['urls'].append({'url': normalize_url(raw_url), 'kind': kind, 'pos': len(result['tokens'])})

    position = 0
    for match in LINK_SYNTAX_RE.finditer(text):
        add_tokens(text[position:match.start()], position)
        position = match.end()
        groups = match.groupdict()
        if groups['def_target'] is not None:
            add_url(groups['def_target'], 'link')
        elif groups['inline_label'] is not None:
            add_url(parse_link_target(groups['inline_target']), 'link')
            _scan_links(groups['inline_label'], result, in_label=True)
        elif groups['ref_label'] is not None:
            # The URL is recorded from the reference definition
            _scan_links(groups['ref_label'], result, in_label=True)
        elif groups['html_label'] is not None:
            target = next(t for t in (groups['html_target1'], groups['html_target2'], groups['html_target3'])
                          if t is not None)
            add_url(target, 'link')
            _scan_links(groups['html_label'], result, in_label=True)
        elif groups['autolink'] is not None:
            add_url(groups['autolink'], 'bare')
        elif groups['bare_url'] is not None:
            url = trim_bare_url(groups['bare_url'])
            add_url(url, 'bare')
            # Give back the trimmed trailing punctuation
            position = match.start() + len(url)
        else:
            add_url(groups['email'], 'bare')
    add_tokens(text[position:], position)


def compare_urls(original_urls, improved_urls):
    """Compare the URLs of both versions. Returns a list of changes, each {type, original, improved}.

    type is 'removed', 'added' or 'altered' (a removed and an added URL at the same place in the document).
    A URL that only moved is not a change.
    """
    original_list = [u['url'] for u in original_urls]
    improved_list = [u['url'] for u in improved_urls]
    removed_budget = Counter(original_list) - Counter(improved_list)
    added_budget = Counter(improved_list) - Counter(original_list)

    changes = []
    matcher = difflib.SequenceMatcher(None, original_list, improved_list, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        removed = []
        for url in original_list[i1:i2]:
            if removed_budget[url] > 0:
                removed_budget[url] -= 1
                removed.append(url)
        added = []
        for url in improved_list[j1:j2]:
            if added_budget[url] > 0:
                added_budget[url] -= 1
                added.append(url)
        for old_url, new_url in zip(removed, added):
            changes.append({'type': 'altered', 'original': old_url, 'improved': new_url})
        for old_url in removed[len(added):]:
            changes.append({'type': 'removed', 'original': old_url, 'improved': None})
        for new_url in added[len(removed):]:
            changes.append({'type': 'added', 'original': None, 'improved': new_url})
    return changes


def is_link_text_edit(mismatch, original_analysis, improved_analysis, converted_url_positions):
    """True if every changed token of the mismatch is an expected link text edit.

    For each changed part: every improved-side token must be inside a link label, and the original-side
    tokens must either all be inside a link label or end directly before a bare URL that became a link.
    """
    original_in_label = original_analysis['in_label']
    improved_in_label = improved_analysis['in_label']
    for i1, i2, j1, j2 in mismatch['parts']:
        if not all(improved_in_label[j1:j2]):
            return False
        if not all(original_in_label[i1:i2]) and i2 not in converted_url_positions:
            return False
    return True
