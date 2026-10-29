"""Image placeholders and image references."""
import re

import pytest
from PIL import Image

TEXT = ('Intro ![Logo](media/image1.png){width="2in"} and ![](media/b.jpg "t") end\n\n'
        'See ![ref][r] too.\n')


def test_round_trip(api):
    protected, placeholders, prefix = api.protect_images(TEXT)
    assert len(placeholders) == 3
    assert prefix not in TEXT
    assert '![' not in protected
    for placeholder in placeholders:
        assert re.fullmatch(re.escape(prefix) + r'\d+X', placeholder)
        assert protected.count(placeholder) == 1
    assert list(placeholders.values()) == ['![Logo](media/image1.png){width="2in"}', '![](media/b.jpg "t")',
                                           '![ref][r]']
    assert api.check_image_placeholders(protected, placeholders, prefix) == []
    restored = api.restore_images(protected, placeholders)
    assert restored == TEXT
    assert api.check_restored_images(TEXT, restored) == []


def test_text_without_images(api):
    protected, placeholders, prefix = api.protect_images('No images here.')
    assert prefix.startswith('IMAGE')
    assert protected == 'No images here.'
    assert placeholders == {}
    assert api.restore_images('No images here.', {}) == 'No images here.'


def test_missing_placeholder(api):
    protected, placeholders, prefix = api.protect_images(TEXT)
    first = f"{prefix}1X"
    problems = api.check_image_placeholders(protected.replace(first, ''), placeholders, prefix)
    assert problems == [f'{first} is missing (was: ![Logo](media/image1.png){{width="2in"}})']


def test_duplicated_placeholder(api):
    protected, placeholders, prefix = api.protect_images(TEXT)
    second = f"{prefix}2X"
    problems = api.check_image_placeholders(protected + ' ' + second, placeholders, prefix)
    assert problems == [f'{second} appears 2 times (was: ![](media/b.jpg "t"))']


def test_altered_placeholder(api):
    protected, placeholders, prefix = api.protect_images(TEXT)
    first = f"{prefix}1X"
    problems = api.check_image_placeholders(protected.replace(first, f"{prefix}one"), placeholders, prefix)
    assert len(problems) == 2
    assert problems[0].startswith(f"{first} is missing")
    assert problems[1].startswith("altered placeholder near: ")
    assert f"{prefix}one" in problems[1]


def test_unknown_placeholder_number(api):
    protected, placeholders, prefix = api.protect_images(TEXT)
    problems = api.check_image_placeholders(protected + f" {prefix}99X", placeholders, prefix)
    assert problems == [f"unknown or altered placeholder {prefix}99X appears 1 time(s)"]


def test_restored_images_wrapped_or_dropped(api):
    original = 'A ![](a.png) B'
    assert api.check_restored_images(original, 'A B') == [
        'image reference missing or changed after restoring (1x): ![](a.png)']
    assert api.check_restored_images(original, 'A ![](a.png) ![x](b.png) B') == [
        'unexpected image reference after restoring (1x): ![x](b.png)']


def test_image_alt_text(api):
    assert api.image_alt_text('![A [b] c](x.png)') == 'A [b] c'
    assert api.image_alt_text('![](x.png)') == ''


def test_image_target_path(api):
    assert api.image_target_path('![](media/a%20b.png "T")') == ('media/a%20b.png', None)
    assert api.image_target_path('![](<my pic.png>)') == ('my pic.png', None)
    # Pandoc writes paths with spaces unbracketed, e.g. the media folder named after the output file
    assert api.image_target_path('![](Oct News_improved_media/media/image1.jpeg){width="1in"}') == (
        'Oct News_improved_media/media/image1.jpeg', None)
    assert api.image_target_path("![](my pic.png 'Title')") == ('my pic.png', None)
    assert api.image_target_path('![](media/a\\_b.png)') == ('media/a_b.png', None)
    assert api.image_target_path('![](C:\\pics\\a.png)') == ('C:\\pics\\a.png', None)
    assert api.image_target_path('![](https://example.com/a.png)') == (
        None, "it is not a local file (https://example.com/a.png); images are not downloaded")
    assert api.image_target_path('![x][r]') == (
        None, "it is a reference-style image (![alt][ref]); looking up its file is not supported")


def test_resolve_image_file(api, tmp_path):
    (tmp_path / 'a b.png').write_bytes(b'x')
    assert api.resolve_image_file('a%20b.png', str(tmp_path)) == str(tmp_path / 'a b.png')
    assert api.resolve_image_file('a b.png', str(tmp_path)) == str(tmp_path / 'a b.png')
    assert api.resolve_image_file('missing.png', str(tmp_path)) is None


def test_image_media_type(api, tmp_path):
    png = tmp_path / 'pic.dat'
    Image.new('RGB', (2, 2)).save(png, format='PNG')
    assert api.image_media_type(str(png)) == 'image/png'
    text = tmp_path / 'not-an-image.png'
    text.write_text('hello')
    with pytest.raises(ValueError, match='not an image format Pillow can identify'):
        api.image_media_type(str(text))
