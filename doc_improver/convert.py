"""Converting input documents to markdown (pandoc) or CSV text (pandas)."""
import os
import re

import pandas as pd
import pypandoc
from colorama import Fore, Style

from .md_syntax import IMAGE_RE


def media_folder_name(html_output):
    """Return the name of the output's own media folder: the output file's name plus _media.

    Each output gets its own folder, so documents converted into the same folder never overwrite each
    other's images (pandoc names a .docx's images image1, image2, ... in every document).
    """
    return os.path.splitext(os.path.basename(html_output))[0] + '_media'


# Pandoc replaces an image it can't read with a placeholder span (or figure) that keeps the original path:
# []{.image .placeholder original-image-src="pics/a.png" ...} or data-original-image-src="..." in raw HTML
IMAGE_PLACEHOLDER_SRC_RE = re.compile(r'original-image-src="([^"]*)"')


def flatten_extracted_media(markdown_text, output_dir, staging_name, media_name):
    """Move the images pandoc extracted into staging_name (inside output_dir) into media_name, dropping the one
    folder level pandoc copies from the input when every image sits under it (a .docx keeps its images in
    media/, so pandoc writes <staging>/media/image1.jpeg). Returns the markdown with each image link
    rewritten to the image's new place; links to anything else are left alone.

    Images from an earlier run of the same output are replaced by name; other files in media_name are kept.
    The emptied staging folder is removed.
    """
    staging_dir = os.path.join(output_dir, staging_name)
    if not os.path.isdir(staging_dir):
        return markdown_text
    extracted = []
    for root, _dirs, files in os.walk(staging_dir):
        for name in files:
            extracted.append(os.path.relpath(os.path.join(root, name), staging_dir).replace(os.sep, '/'))
    top_dirs = {path.split('/', 1)[0] for path in extracted}
    strip = len(top_dirs) == 1 and all('/' in path for path in extracted)
    moves = {}
    for path in extracted:
        new_path = path.split('/', 1)[1] if strip else path
        source = os.path.join(staging_dir, *path.split('/'))
        target = os.path.join(output_dir, media_name, *new_path.split('/'))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moves[f"{staging_name}/{path}"] = f"{media_name}/{new_path}"
    for root, _dirs, _files in os.walk(staging_dir, topdown=False):
        os.rmdir(root)

    def rewrite(match):
        image_ref = match.group(0)
        for old, new in moves.items():
            image_ref = image_ref.replace(old, new)
        return image_ref
    return IMAGE_RE.sub(rewrite, markdown_text)


def find_missing_images(pandoc_markdown):
    """Return the path of every image pandoc could not find (one entry per occurrence, in document order)."""
    return IMAGE_PLACEHOLDER_SRC_RE.findall(pandoc_markdown)


def print_missing_images(missing, input_dir):
    """Print a loud warning naming each image pandoc could not find and replaced with a placeholder."""
    print(Fore.RED + Style.BRIGHT + f"\nWARNING: pandoc could not find {len(missing)} image(s) in the input. "
          "Each was replaced with a text placeholder, so it will be missing from the output:")
    for path in missing:
        print(Fore.RED + f"  - {path} (relative paths are looked up in {input_dir})")


def convert_to_markdown(input_path, output_path, html_output, verbose=False):
    """Convert a supported document file to markdown using pypandoc.

    Images are extracted into the output's own media folder (media_folder_name) next to html_output, and the
    markdown references them relative to html_output's folder, so the image links work from the HTML and the
    improved markdown saved beside it. Pandoc keeps a .docx's internal media/ folder (it has no option to leave
    that level out), so it extracts into a staging folder and flatten_extracted_media moves the images up,
    e.g. to <name>_media/image1.jpeg, and rewrites the image links to match.
    Relative image paths in the input (e.g. a .md or .html file) are looked up relative to the input file.
    Images pandoc can't find are listed in a printed warning. verbose prints a line when the conversion is done.
    """
    output_dir = os.path.dirname(os.path.abspath(html_output))
    input_dir = os.path.dirname(os.path.abspath(input_path))
    # --resource-path is a list separated by os.pathsep, so a folder whose name contains it can't be given
    if os.pathsep in input_dir:
        raise ValueError(f"the input folder {input_dir} contains '{os.pathsep}', which pandoc reads as a "
                         "separator in its image search path; move the input to a folder without it")
    # Set extra arguments for pandoc
    media_name = media_folder_name(html_output)
    staging_name = media_name + '.extracting'
    extra_args = ["--wrap=none", f"--extract-media={staging_name}", f"--resource-path={input_dir}"]

    # Use pypandoc to convert the file to markdown. Pandoc runs inside the output folder so the relative
    # media path is created and referenced there; input/output paths are made absolute for that reason.
    pypandoc.convert_file(os.path.abspath(input_path), 'markdown', outputfile=os.path.abspath(output_path),
                          extra_args=extra_args, cworkdir=output_dir)

    if verbose:
        print(f"Successfully converted {input_path} to {output_path}")
    
    # Read the markdown content
    with open(output_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    markdown_content = flatten_extracted_media(markdown_content, output_dir, staging_name, media_name)

    missing_images = find_missing_images(markdown_content)
    if missing_images:
        print_missing_images(missing_images, input_dir)
        
    # Convert pandoc markdown to standard markdown
    # (no paragraph unwrapping needed: pandoc runs with --wrap=none)
    standard_markdown = convert_pandoc_markdown_to_standard(markdown_content)
    
    # Write back the processed content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(standard_markdown)
    
    return standard_markdown


def get_csv_content_from_spreadsheet(input_path, verbose=False):
    """Convert spreadsheet (XLS, XLSX) to a CSV string using pandas. verbose prints a line when it is done."""
    # Only single-sheet workbooks are supported; refuse rather than silently dropping other sheets
    with pd.ExcelFile(input_path) as workbook:
        sheet_names = workbook.sheet_names
        if len(sheet_names) > 1:
            raise ValueError(f"{input_path} has {len(sheet_names)} sheets ({', '.join(map(str, sheet_names))}). "
                             "Only single-sheet workbooks are supported; split the sheets into separate files.")
        df = pd.read_excel(workbook, sheet_name=0)
    
    if verbose:
        print(f"Successfully extracted data from {input_path}")
    
    # Return the content of the DataFrame as a CSV string
    return df.to_csv(index=False)


def convert_pandoc_markdown_to_standard(markdown_text):
    """
    Convert Pandoc-specific markdown to more standard markdown that GPT can understand.
    """
    # Replace [text]{.underline} with **text** (must run before the bare {.underline} removal below).
    # The text may contain one level of brackets, e.g. pandoc's underlined links: [[text](url)]{.underline}
    markdown_text = re.sub(r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\{\s*\.underline\s*\}', r'**\1**', markdown_text)

    # Remove any remaining pandoc {.underline} attributes
    markdown_text = re.sub(r'\{\s*\.underline\s*\}', '', markdown_text)
    
    # Clean up heading brackets - pattern **[Heading]** or *[Heading]**
    markdown_text = re.sub(r'\*+\s*\[(.*?)\]\s*\*+', r'**\1**', markdown_text)
    
    # Clean up heading brackets in potential heading lines.
    # Skip markdown reference-link definitions like "[1]: https://example.com" or '[x]: <url> "title"'.
    ref_definition = r'''[ \t]*(?:<[^>\n]*>|\S+)(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*$'''
    pattern = r'^(\s*)\[(.*?)\]:(?!' + ref_definition + r')(.*?)$'
    replacement = r'\1\2:\3'
    markdown_text = re.sub(pattern, replacement, markdown_text, flags=re.MULTILINE)
    
    # Remove other pandoc-specific class attributes
    markdown_text = re.sub(r'\{\s*\.[a-zA-Z0-9_-]+\s*\}', '', markdown_text)
    
    return markdown_text


def convert_markdown_to_html(markdown_text):
    """Return the markdown converted to a standalone HTML page with pandoc."""
    # --standalone adds <head> and <meta charset="utf-8">. pagetitle (not title) sets <title>
    # without pandoc inserting a visible "Document" heading at the top of the body.
    extra_args = ["--standalone", "--metadata", "pagetitle=Document"]
    # Use a format string that explicitly enables pipe tables, which is what our fallback uses.
    # -implicit_figures: otherwise an image alone in its paragraph becomes a <figure> whose alt text is
    # also shown as a visible <figcaption>; alt text should only be read by screen readers.
    return pypandoc.convert_text(markdown_text, 'html', format='markdown-native_divs+pipe_tables-implicit_figures',
                                 extra_args=extra_args)
