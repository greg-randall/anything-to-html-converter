"""Markdown syntax shared by the other modules: image, heading, attribute and link patterns, and the tokenizer."""
import re

# A markdown image reference: ![alt](target "title") or ![alt][ref], with an optional pandoc
# attribute block such as {width="2in" height="1in"}. The alt text may contain one level of
# brackets and the target one level of parentheses; backslash escapes are allowed in both.
IMAGE_RE = re.compile(
    r'!\[(?:[^\[\]\\]|\\.|\[(?:[^\[\]\\]|\\.)*\])*\]'
    r'(?:\((?:[^()\\\n]|\\.|\((?:[^()\\\n]|\\.)*\))*\)|\[(?:[^\[\]\\\n]|\\.)*\])'
    r'(?:\{[^{}\n]*\})?'
)


# Start of an image reference with its alt text: ![alt] (the alt may contain one level of brackets)
IMAGE_ALT_RE = re.compile(r'!\[((?:[^\[\]\\]|\\.|\[(?:[^\[\]\\]|\\.)*\])*)\]')
# Inline image target right after the alt: (target "optional title")
IMAGE_TARGET_RE = re.compile(r'\(((?:[^()\\\n]|\\.|\((?:[^()\\\n]|\\.)*\))*)\)')
# An ATX heading line (# to ######); its level is the number of #
HEADING_RE = re.compile(r'^ {0,3}(#{1,6})(?:[ \t]|$)')
# A code fence line (``` or ~~~), inside which # lines are not headings
FENCE_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})')


def find_headings(markdown_text):
    """Return the ATX heading lines of the text as (offset, level, text), skipping lines inside code fences."""
    headings = []
    offset = 0
    fence = None
    for line in markdown_text.splitlines(keepends=True):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        elif fence is None:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                headings.append((offset, len(heading_match.group(1)), line.strip()))
        offset += len(line)
    return headings


# One item of a pandoc attribute block: #id, .class, key=value / key="value", =format (raw attribute) or "-"
_ATTRIBUTE_ITEM = r"""(?:#[\w:.-]+|\.[\w-]+|[\w-]+=(?:"[^"\n]*"|'[^'\n]*'|[^\s{}"']+)|=[\w-]+|-)"""
_ATTRIBUTE_BLOCK = r'\{[ \t]*' + _ATTRIBUTE_ITEM + r'(?:[ \t]+' + _ATTRIBUTE_ITEM + r')*[ \t]*\}'
# Pandoc attribute blocks sit right after a link/image/span/code (`]`, `)` or a backtick), or at the
# end of a line (headings, fenced divs and code fences)
ATTRIBUTE_BLOCK_RE = re.compile(r'(?<=[\])`])' + _ATTRIBUTE_BLOCK + r'|' + _ATTRIBUTE_BLOCK + r'(?=[ \t]*$)',
                                re.MULTILINE)
# Pandoc superscript ^text^ and subscript ~text~ (no unescaped spaces inside; ~~strikeout~~ is not subscript)
SUPERSCRIPT_RE = re.compile(r'\^((?:[^\s^\\]|\\.)+)\^')
SUBSCRIPT_RE = re.compile(r'(?<!~)~((?:[^\s~\\]|\\.)+)~(?!~)')
# HTML tags and comments (an autolink like <https://...> or <a@b.c> is not a tag: no ':' or '@' in a tag name)
HTML_TAG_RE = re.compile(r'<!--.*?-->|</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>', re.DOTALL)
# A word: a run of letters/digits in any script (underscores and punctuation separate words)
WORD_RE = re.compile(r'[^\W_]+')


def strip_inline_markup(text):
    """Remove superscript/subscript markers and HTML tags so only the visible words remain.

    Markers are removed without a space, so 100^th^ reads as 100th and H~2~O as H2O.
    HTML tags are replaced with a space so they never join two words.
    """
    text = SUPERSCRIPT_RE.sub(r'\1', text)
    text = SUBSCRIPT_RE.sub(r'\1', text)
    return HTML_TAG_RE.sub(' ', text)


def tokenize_text(text):
    """
    Tokenize plain text (no links or URLs) into words and numbers, ignoring inline markup.
    """
    return WORD_RE.findall(strip_inline_markup(text))


# Link label: may contain one level of brackets and backslash escapes
_LINK_LABEL = r'(?:[^\[\]\\]|\\.|\[(?:[^\[\]\\]|\\.)*\])*'
# Inline link target: may contain one level of parentheses and backslash escapes
_LINK_TARGET = r'(?:[^()\\\n]|\\.|\((?:[^()\\\n]|\\.)*\))*'
# Everything that carries a URL, in markdown or HTML syntax. Alternatives are tried in this order at each position.
LINK_SYNTAX_RE = re.compile(
    # Reference definition on its own line: [label]: url "optional title"
    r'(?m:^[ \t]{0,3}\[(?P<def_label>[^\[\]\n]+)\]:[ \t]*(?P<def_target><[^>\n]*>|\S+)'
    r'''(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*$)'''
    # Inline link: [label](target "optional title")
    r'|\[(?P<inline_label>' + _LINK_LABEL + r')\]\((?P<inline_target>' + _LINK_TARGET + r')\)'
    # Full reference link: [label][ref]
    r'|\[(?P<ref_label>' + _LINK_LABEL + r')\]\[(?P<ref_id>[^\[\]\n]*)\]'
    # HTML link: <a href="url">label</a>
    r'''|(?i:<a\s[^>]*?href\s*=\s*(?:"(?P<html_target1>[^"]*)"|'(?P<html_target2>[^']*)'|(?P<html_target3>[^\s>]+))'''
    r'[^>]*>(?P<html_label>[\s\S]*?)</a\s*>)'
    # Autolink: <scheme:...> or <user@host>
    r'|<(?P<autolink>[A-Za-z][A-Za-z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>'
    # Bare URL with a scheme (scheme://... or mailto:...)
    r'|(?<![\w/])(?P<bare_url>[A-Za-z][A-Za-z0-9+.-]*://[^\s<>]+|(?i:mailto):[^\s<>]+)'
    # Bare email address
    r'|(?<![\w.+-])(?P<email>[\w.+-]+@[^\W_](?:[\w-]*[^\W_])?(?:\.[^\W_](?:[\w-]*[^\W_])?)+)'
)
URL_SCHEME_RE = re.compile(r'[A-Za-z][A-Za-z0-9+.-]*:')
