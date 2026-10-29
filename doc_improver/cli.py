"""The command line: argument parsing and the main conversion pipeline."""
import argparse
import difflib
import os

from colorama import Fore, Style, init

# Importing these modules loads .env (in doc_improver.config), so the argument defaults in build_parser
# (os.getenv) see its settings
from .alt_text import finish_alt_text, restore_images_with_alt_text, start_alt_text
from .compare import compare_texts
from .config import (
    API_STYLE_NAMES,
    get_alt_text_settings,
    get_connection,
    resolve_format_settings,
)
from .convert import (
    convert_markdown_to_html,
    convert_to_markdown,
    get_csv_content_from_spreadsheet,
)
from .formatting import format_content_with_llm
from .images import (
    check_image_placeholders,
    check_restored_images,
    image_alt_text,
    protect_images,
    restore_images,
)
from .pricing import (
    cost_summary,
    estimate_cost,
    load_prices,
    print_cost_breakdown,
    print_cost_total,
)
from .report import (
    format_tokens,
    plural,
    print_added_alt_text_report,
    print_link_text_edits,
    print_list_numbering,
    print_mismatch_report,
    print_missing_and_added_words,
    print_preserved_summary,
    print_url_report,
)

# Initialize colorama to auto-reset colors after each print
init(autoreset=True)


def build_parser():
    """Return the argument parser. The defaults read .env settings, which are loaded when config is imported."""
    parser = argparse.ArgumentParser(description='Convert documents to Markdown and HTML and improve them with an '
                                     'LLM (any endpoint that speaks the Anthropic or OpenAI API)')
    parser.add_argument('input_file', help='Path to input file (e.g., .docx, .xlsx, .html, .rtf)')
    parser.add_argument('--output', help='Path to the output .html (or .md) file. Both the HTML and the improved '
                        'markdown are saved, with the same name and the other extension. '
                        'Default: {input}_improved.html')
    parser.add_argument('--api', choices=API_STYLE_NAMES, default=os.getenv('LLM_API'),
                        help='The API style the endpoint speaks (default: LLM_API from .env)')
    parser.add_argument('--base-url', default=os.getenv('LLM_BASE_URL'),
                        help="The endpoint's base URL (default: LLM_BASE_URL from .env; if unset, the API's "
                        'official endpoint)')
    parser.add_argument('--model', help='Model for formatting (default: FORMAT_MODEL from .env)')
    parser.add_argument('--max-tokens', type=int,
                        help='Maximum output tokens for formatting (default: FORMAT_MAX_TOKENS from .env)')
    parser.add_argument('--effort',
                        help='Effort level for formatting: Anthropic effort or OpenAI reasoning_effort '
                        '(default: FORMAT_EFFORT from .env; if unset, the API default applies)')
    parser.add_argument('--no-alt-text', dest='alt_text', action='store_false',
                        help='Do not generate alt text for images that have none')
    parser.add_argument('--alt-text-model',
                        help='Model for alt text; it must accept images (default: ALT_TEXT_MODEL from .env; if '
                        'unset, the formatting model)')
    parser.add_argument('--alt-text-max-tokens', type=int,
                        help='Maximum output tokens (thinking included) for each alt text call '
                        '(default: ALT_TEXT_MAX_TOKENS from .env)')
    parser.add_argument('--alt-text-effort',
                        help='Effort level for the alt text calls (default: ALT_TEXT_EFFORT from .env; if unset, '
                        'the API default applies)')
    parser.add_argument('--alt-text-context-words', type=int,
                        help='Minimum number of words of surrounding text sent before and after each image, '
                        'in whole paragraphs (default: ALT_TEXT_CONTEXT_WORDS from .env)')
    parser.add_argument('--alt-text-concurrency', type=int, metavar='N',
                        help='Maximum number of alt text calls running at once; they run alongside the formatting '
                        'call, which is not counted (default: ALT_TEXT_CONCURRENCY from .env)')
    parser.add_argument('--show-diff', action='store_true', help='Show diff between original and improved markdown')
    parser.add_argument('--no-detailed-comparison', dest='detailed_comparison', action='store_false',
                        help='Disable detailed token-by-token comparison')
    parser.set_defaults(detailed_comparison=True)
    parser.add_argument('--keep-markdown', action='store_true', help='Keep the intermediate markdown files')
    parser.add_argument('--verbose', action='store_true',
                        help='Show the details: endpoint and model of each model call, full token '
                        'usage, the text sent with each image for alt text, each generated alt text as it arrives, '
                        'and words added by the model. Without it the output is a short summary; warnings, errors '
                        'and content differences are always shown in full.')
    return parser


def parse_settings():
    """Parse the command line and fill in the .env settings. Returns (args, parser, connection)."""
    parser = build_parser()
    args = parser.parse_args()
    connection = get_connection(args, parser)
    resolve_format_settings(args, parser)
    return args, parser, connection


def resolve_paths(args):
    """Return (intermediate_md_path, html_output, improved_md_output), or None (the error is printed)."""
    input_file = args.input_file
    base_name = os.path.splitext(input_file)[0]

    intermediate_md_path = f"{base_name}.md"
    # Never write to (or later delete) the input file itself, e.g. when the input is already .md
    if os.path.abspath(intermediate_md_path) == os.path.abspath(input_file):
        intermediate_md_path = f"{base_name}_converted.md"

    # --- DETERMINE OUTPUT PATHS ---
    # The final output is HTML plus the improved markdown, saved side by side.
    # --output picks the path of one of them; the other gets the same name with the other extension.
    output_path = args.output if args.output else f"{base_name}_improved.html"
    output_base, output_ext = os.path.splitext(output_path)
    if output_ext.lower() in ('.html', '.htm'):
        html_output = output_path
        improved_md_output = f"{output_base}.md"
    elif output_ext.lower() == '.md':
        improved_md_output = output_path
        html_output = f"{output_base}.html"
    else:
        print(f'Error: Invalid output format! Got "{output_ext}" but --output must end in .html, .htm or .md')
        return
    for path in (html_output, improved_md_output):
        if os.path.abspath(path) in (os.path.abspath(input_file), os.path.abspath(intermediate_md_path)):
            print(f"Error: Output path {path} would overwrite the input file or the intermediate markdown file.")
            print("Choose a different --output.")
            return
    return intermediate_md_path, html_output, improved_md_output


# Spreadsheets are read with pandas; everything else is sent to pandoc, which decides what it can read
SPREADSHEET_EXTENSIONS = {'.xlsx', '.xls'}


def is_spreadsheet(input_file):
    return os.path.splitext(input_file)[1].lower() in SPREADSHEET_EXTENSIONS


def extract_content(input_file, intermediate_md_path, html_output, verbose=False):
    """Return the input as markdown (via pandoc) or CSV text (spreadsheets), or None (the error is printed)."""
    # --- GET INITIAL CONTENT (EITHER MARKDOWN OR CSV) ---
    original_content = None
    if is_spreadsheet(input_file):
        if verbose:
            print(f"Detected spreadsheet: {input_file}. Extracting CSV content to pass to AI...")
        original_content = get_csv_content_from_spreadsheet(input_file, verbose=verbose)
    else:
        if verbose:
            print(f"Detected document: {input_file}. Converting to Markdown...")
        try:
            original_content = convert_to_markdown(input_file, intermediate_md_path, html_output, verbose=verbose)
        except RuntimeError as e:
            # pypandoc raises RuntimeError when pandoc can't read the input; show pandoc's own message
            print(f"Error: pandoc could not convert {input_file}:\n{e}")
            print("Conversion failed. Exiting.")
            return
        except ValueError as e:
            print(f"Error: {e}")
            print("Conversion failed. Exiting.")
            return

    if original_content is None:
        print("Initial content extraction or conversion failed. Exiting.")
        return
    return original_content


def format_content(args, parser, connection, original_content, html_output, prices, costs):
    """Protect the images, start the alt text calls, and format the content with the LLM while they run.

    The alt text calls (for images without alt text) start before the formatting call and run in parallel with
    it and with each other; their context comes from original_content. After the formatting call, the alt text
    results are collected and printed. If the formatting call failed, the alt text calls not started yet are
    cancelled and the rest are waited for. The estimated cost of every finished call (pricing.CallCost, priced
    with prices) is appended to costs, whether or not the formatting call succeeded.

    Returns (final_markdown, image_placeholders, placeholder_prefix, alt_text_result), or None if the formatting
    call failed (the error is printed). final_markdown still has the image placeholders. alt_text_result is an
    alt_text.AltTextResult, or None when no alt text was generated.
    """
    # Replace image references with placeholders so the model can't alter them (sizes, alt text, paths)
    protected_content, image_placeholders, placeholder_prefix = protect_images(original_content)
    if args.verbose:
        if image_placeholders:
            print(f"Protected {len(image_placeholders)} image reference(s) with placeholders.")
    elif is_spreadsheet(args.input_file):
        print(f"Extracting {args.input_file} as CSV... done")
    else:
        images = (plural(len(image_placeholders), 'image') + ' protected') if image_placeholders else 'no images'
        print(f"Converting {args.input_file} to markdown... done ({images})")

    # Alt text is generated for images without alt text. Check its settings now, before any call, so a missing
    # setting doesn't waste a call.
    alt_text_settings = None
    if args.alt_text and any(not image_alt_text(ref).strip() for ref in image_placeholders.values()):
        alt_text_settings = get_alt_text_settings(args)
        if isinstance(alt_text_settings, str):
            parser.error(alt_text_settings)

    alt_text_run = None
    if alt_text_settings:
        alt_text_run = start_alt_text(original_content, image_placeholders,
                                      os.path.dirname(os.path.abspath(html_output)), connection, alt_text_settings,
                                      prices, verbose=args.verbose)

    # --- Format content with the LLM, failing on error as requested ---
    if args.verbose:
        print(f"Formatting content with {args.model} (max tokens {args.max_tokens})")
    label = f"Formatting with {args.model} (effort {args.effort or 'API default'})"
    prompt_placeholder_prefix = placeholder_prefix if image_placeholders else None
    try:
        final_markdown, tokens = format_content_with_llm(protected_content, connection=connection,
                                                         model=args.model, max_tokens=args.max_tokens,
                                                         effort=args.effort,
                                                         image_placeholder_prefix=prompt_placeholder_prefix,
                                                         label=label, verbose=args.verbose)
    except BaseException:
        # e.g. Ctrl+C or a missing setting: don't leave alt text calls running behind the exception
        if alt_text_run:
            alt_text_run.abort()
        raise
    cost = estimate_cost(args.model, tokens, prices)
    costs.append(cost)
    print(f"{label}... {'done' if final_markdown else 'failed'}: {format_tokens(tokens)}{cost_summary([cost])}")
    if args.verbose:
        print_cost_breakdown([cost])

    alt_text_result = None
    if alt_text_run:
        stop_reason = None
        if not final_markdown:
            # The alt text can't be used: cancel the calls not started yet; the running ones finish and are priced
            alt_text_run.cancel_pending()
            stop_reason = "the formatting call failed"
        alt_text_result = finish_alt_text(alt_text_run, stop_reason=stop_reason)
        costs.extend(alt_text_result.costs)

    # If the API call failed, exit the program as requested.
    if not final_markdown:
        print("\n--- FATAL ERROR ---")
        print(f"The formatting call to {args.model} failed and no formatted content was returned.")
        print("The script will not produce a fallback HTML file and will now exit.")
        print("Please check the API error message above, your configuration, endpoint status, and permissions.")
        print("---------------------")
        return None
    if not args.alt_text:
        blank_line = "\n" if args.verbose else ""
        print(f"{blank_line}Alt text generation is off (--no-alt-text).")
    return final_markdown, image_placeholders, placeholder_prefix, alt_text_result


def restore_images_and_insert_alt_text(original_content, final_markdown, image_placeholders, placeholder_prefix,
                                       alt_text_result):
    """Put the image references back and insert the generated alt text. Returns (final_markdown, generated,
    warnings), or None if the image placeholders were not preserved (the problems are printed and nothing is
    written)."""
    # Put the original image references back, failing loudly (and writing nothing) if the model
    # dropped, duplicated or altered any placeholder
    image_problems = check_image_placeholders(final_markdown, image_placeholders, placeholder_prefix)
    if not image_problems:
        restored_markdown = restore_images(final_markdown, image_placeholders)
        image_problems = check_restored_images(original_content, restored_markdown)
    if image_problems:
        print(Fore.RED + Style.BRIGHT + "\n--- FATAL ERROR: image placeholders were not preserved ---")
        for problem in image_problems:
            print(Fore.RED + f"  - {problem}")
        print(Fore.RED + "No output was written.")
        return None
    if not alt_text_result:
        return restored_markdown, [], []

    # Only now that the restored references match the originals: restore again, this time with each generated
    # alt text in its own image's reference (paired by placeholder, so moved images keep the right alt text).
    # Author-written alt text is never changed. The comparison below skips image references, so generated alt
    # text never counts as changed content.
    final_markdown = restore_images_with_alt_text(final_markdown, image_placeholders,
                                                  alt_text_result.alt_by_placeholder)
    return final_markdown, alt_text_result.generated, alt_text_result.warnings


def write_outputs(final_markdown, improved_md_output, html_output, verbose=False):
    """Save the improved markdown and its HTML conversion."""
    # Save the improved markdown
    with open(improved_md_output, 'w', encoding='utf-8') as f:
        f.write(final_markdown)
    if verbose:
        print(f"\nImproved markdown saved to {improved_md_output}")

    # Always convert the final markdown to HTML
    html_content = convert_markdown_to_html(final_markdown)
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    if verbose:
        print(f"\nHTML conversion complete. Final output saved to {html_output}")
    else:
        print(f"Saved {improved_md_output} and {html_output}")


def compare_and_report(args, original_content, final_markdown):
    """Compare the original and improved text and print the URL, word and difference reports."""
    # Compare texts to ensure no content was lost
    comparison_result = compare_texts(original_content, final_markdown)

    # By default, what was preserved goes on one line; problems are always printed in full below
    if not args.verbose:
        print_preserved_summary(comparison_result)
    # A removed, added or altered URL is a real error; report it loudly (output is still written)
    if args.verbose or comparison_result['url_changes']:
        print_url_report(comparison_result['url_changes'])
    print_missing_and_added_words(comparison_result, args.verbose)

    # Expected link label rewrites, listed briefly and separately from the content differences
    print_link_text_edits(comparison_result, verbose=args.verbose)
    # Numbers added to or removed from numbered lists, and lists whose numbers are out of order
    print_list_numbering(comparison_result, verbose=args.verbose)

    unexplained_count = sum(1 for m in comparison_result['mismatches']
                            if not (m['link_text_edit'] or m['list_numbering']))
    if args.detailed_comparison and unexplained_count:
        print_mismatch_report(comparison_result, detailed=True)
    elif unexplained_count:
        print(f"\nFound {unexplained_count} content differences.")
        print("To see details, run without --no-detailed-comparison")

    if args.show_diff:
        print("\n===== UNIFIED LINE DIFF =====")
        diff = difflib.unified_diff(
            original_content.splitlines(),
            final_markdown.splitlines(),
            lineterm='',
            n=3
        )
        print('\n'.join(diff))


def main():
    """Run the pipeline: settings, paths, extract, format and generate alt text in parallel, restore images
    and insert the alt text, write, report."""
    args, parser, connection = parse_settings()
    # Read prices.yaml before any model call, so a malformed file stops the run before money is spent
    prices = load_prices()
    # The estimated cost of every model call in the run (pricing.CallCost), for the total line
    costs = []

    paths = resolve_paths(args)
    if paths is None:
        return
    intermediate_md_path, html_output, improved_md_output = paths

    original_content = extract_content(args.input_file, intermediate_md_path, html_output, verbose=args.verbose)
    if original_content is None:
        return

    formatted = format_content(args, parser, connection, original_content, html_output, prices, costs)
    if formatted is None:
        print_cost_total(costs, verbose=args.verbose)
        return
    final_markdown, image_placeholders, placeholder_prefix, alt_text_result = formatted

    restored = restore_images_and_insert_alt_text(original_content, final_markdown, image_placeholders,
                                                  placeholder_prefix, alt_text_result)
    # The total comes right after the model calls; it is labeled once as an estimate at list price
    print_cost_total(costs, verbose=args.verbose)
    if restored is None:
        return
    final_markdown, alt_text_generated, alt_text_warnings = restored

    write_outputs(final_markdown, improved_md_output, html_output, verbose=args.verbose)
    compare_and_report(args, original_content, final_markdown)

    # Clean up intermediate markdown from pandoc conversion if it exists and is not requested
    if not args.keep_markdown and os.path.exists(intermediate_md_path):
        os.remove(intermediate_md_path)

    # Last thing on screen: text the model ADDED to the document (alt text), and images still without it
    print_added_alt_text_report(alt_text_generated, alt_text_warnings)
