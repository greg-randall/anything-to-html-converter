"""Finding the ordered (numbered) lists of a markdown text, so their numbers can be told apart from words."""
import re
from itertools import pairwise

from .md_syntax import FENCE_RE

# An ordered list item: indent, the number, then "." or ")" and a space (or the end of the line).
# An escaped number such as 1\. is plain text, not a list item.
ORDERED_MARKER_RE = re.compile(r'^([ \t]*)(\d{1,9})([.)])(?=[ \t]|$)')
# A bulleted list item: indent, then -, * or + and a space
BULLET_MARKER_RE = re.compile(r'^([ \t]*)[-*+](?=[ \t]|$)')
HEADING_LINE_RE = re.compile(r'^ {0,3}#{1,6}(?:[ \t]|$)')


def indent_width(indent):
    """Width of leading whitespace, counting a tab as 4 columns."""
    return len(indent.expandtabs(4))


def find_ordered_lists(text):
    """Return the ordered lists of text, in order of their first item.

    Each list is {'numbers': [int, ...], 'offsets': [offset of each number in text, ...], 'consecutive': bool};
    consecutive is True when each number is one more than the one before (a list may start at any number).
    Items at a deeper indent form their own list inside the outer one. A list ends at a heading, at a bullet
    or a differently delimited item at its indent or less, or at an unindented line after a blank line (pandoc
    separates lists with <!-- -->). Lines inside code fences are skipped.
    """
    lists = []
    open_lists = []  # (indent, delimiter, list) from outermost to innermost
    offset = 0
    fence = None
    previous_blank = False
    for line in text.splitlines(keepends=True):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
                open_lists = []
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        elif fence is None:
            ordered = ORDERED_MARKER_RE.match(line)
            bullet = BULLET_MARKER_RE.match(line)
            if ordered:
                indent = indent_width(ordered.group(1))
                delimiter = ordered.group(3)
                open_lists = [entry for entry in open_lists if entry[0] <= indent]
                if open_lists and open_lists[-1][0] == indent and open_lists[-1][1] != delimiter:
                    open_lists.pop()
                if not open_lists or open_lists[-1][0] != indent:
                    current = {'numbers': [], 'offsets': []}
                    lists.append(current)
                    open_lists.append((indent, delimiter, current))
                current = open_lists[-1][2]
                current['numbers'].append(int(ordered.group(2)))
                current['offsets'].append(offset + ordered.start(2))
            elif bullet:
                indent = indent_width(bullet.group(1))
                open_lists = [entry for entry in open_lists if entry[0] < indent]
            elif HEADING_LINE_RE.match(line) or (line.strip() and previous_blank and not line[0].isspace()):
                open_lists = []
        previous_blank = not line.strip()
        offset += len(line)
    for found in lists:
        numbers = found['numbers']
        found['consecutive'] = all(b == a + 1 for a, b in pairwise(numbers))
    return lists
