"""Image references: protecting them from the LLM with placeholders, and finding their files."""
import os
import re
import secrets
import urllib.parse
from collections import Counter

from PIL import Image, UnidentifiedImageError

from .md_syntax import IMAGE_ALT_RE, IMAGE_RE, IMAGE_TARGET_RE


def protect_images(markdown_text):
    """Replace every image reference with a unique placeholder so the LLM can't change it.

    Returns (protected_text, placeholders, prefix): placeholders maps each placeholder to the
    original image reference, and prefix is the common start of all placeholders (it includes
    a random part that is checked not to occur anywhere in the text).
    """
    while True:
        prefix = f"IMAGE{secrets.token_hex(4).upper()}N"
        if prefix not in markdown_text:
            break

    placeholders = {}

    def replace(match):
        # The trailing "X" ends the placeholder, so text right after the image can't merge into its number
        placeholder = f"{prefix}{len(placeholders) + 1}X"
        placeholders[placeholder] = match.group(0)
        return placeholder

    protected_text = IMAGE_RE.sub(replace, markdown_text)
    return protected_text, placeholders, prefix


def check_image_placeholders(text, placeholders, prefix):
    """Return a list of problems with the image placeholders in the LLM output (empty if all is well).

    Every placeholder must appear exactly once, and nothing else starting with the prefix may appear.
    """
    problems = []
    found = Counter(re.findall(re.escape(prefix) + r'\d+X', text))
    for placeholder, original in placeholders.items():
        count = found.get(placeholder, 0)
        if count == 0:
            problems.append(f"{placeholder} is missing (was: {original})")
        elif count > 1:
            problems.append(f"{placeholder} appears {count} times (was: {original})")
    for token, count in found.items():
        if token not in placeholders:
            problems.append(f"unknown or altered placeholder {token} appears {count} time(s)")
    # Any other occurrence of the prefix is a placeholder whose number or ending was altered
    for match in re.finditer(re.escape(prefix) + r'(?!\d+X)', text):
        context = text[max(0, match.start() - 20):match.end() + 20]
        problems.append(f"altered placeholder near: {context!r}")
    return problems


def restore_images(text, placeholders):
    """Put the original image references back in place of their placeholders, verbatim."""
    if not placeholders:
        return text
    pattern = re.compile('|'.join(re.escape(p) for p in sorted(placeholders, key=len, reverse=True)))
    return pattern.sub(lambda match: placeholders[match.group(0)], text)


def check_restored_images(original_text, restored_text):
    """Return a list of problems if the image references in restored_text differ from original_text.

    Catches placeholders that the LLM kept but wrapped in other syntax (e.g. inside a new image or link).
    """
    original_images = Counter(IMAGE_RE.findall(original_text))
    restored_images = Counter(IMAGE_RE.findall(restored_text))
    problems = []
    for image, count in (original_images - restored_images).items():
        problems.append(f"image reference missing or changed after restoring ({count}x): {image}")
    for image, count in (restored_images - original_images).items():
        problems.append(f"unexpected image reference after restoring ({count}x): {image}")
    return problems


def image_alt_text(image_ref):
    """Return the alt text of an image reference (as written, with markdown escapes)."""
    return IMAGE_ALT_RE.match(image_ref).group(1)


# An optional link title at the end of an inline target: "title", 'title' or (title), after whitespace
_LINK_TITLE_RE = re.compile(r'''(.*?)(?:\s+(?:"[^"]*"|'[^']*'|\([^()]*\)))?''', re.DOTALL)


def image_link_url(target):
    """Return the URL from an inline image target, dropping an optional title.

    Unlike parse_link_target, spaces are kept: pandoc writes paths with spaces unescaped and unbracketed,
    e.g. ![](Example Document_improved_media/media/image1.jpeg), and reads them back that way.
    """
    target = target.strip()
    if target.startswith('<') and '>' in target:
        return target[:target.index('>') + 1]
    return _LINK_TITLE_RE.fullmatch(target).group(1)


def image_target_path(image_ref):
    """Return (path, problem) for an image reference: the file path it points to, or None and why not."""
    alt_match = IMAGE_ALT_RE.match(image_ref)
    target_match = IMAGE_TARGET_RE.match(image_ref, alt_match.end())
    if not target_match:
        return None, "it is a reference-style image (![alt][ref]); looking up its file is not supported"
    target = image_link_url(target_match.group(1))
    if target.startswith('<') and target.endswith('>'):
        target = target[1:-1]
    # Markdown backslash escapes cover ASCII punctuation only, so a Windows path like C:\pics\a.png is kept
    target = re.sub(r'\\([!-/:-@\[-`{-~])', r'\1', target)
    scheme = urllib.parse.urlparse(target).scheme
    # A one-letter "scheme" is a Windows drive letter (C:\...), not a URL
    if len(scheme) > 1:
        return None, f"it is not a local file ({target}); images are not downloaded"
    return target, None


def resolve_image_file(path, base_dir):
    """Return the existing file for an image path (relative paths are relative to base_dir), or None."""
    candidates = [path, urllib.parse.unquote(path)]
    for candidate in candidates:
        full_path = candidate if os.path.isabs(candidate) else os.path.join(base_dir, candidate)
        if os.path.isfile(full_path):
            return full_path
    return None


def image_media_type(file_path):
    """Return the image's media type (e.g. image/png), read from the file's contents with Pillow.

    Raises ValueError if Pillow can't identify the file or has no media type for its format.
    """
    try:
        with Image.open(file_path) as image:
            image_format = image.format
    except UnidentifiedImageError as e:
        raise ValueError(f"the file is not an image format Pillow can identify ({e})") from e
    media_type = Image.MIME.get(image_format)
    if not media_type:
        raise ValueError(f"no media type is known for image format {image_format}")
    return media_type
