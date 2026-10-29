"""Printing the comparison and alt text reports."""
import re
from itertools import pairwise

from colorama import Fore, Style

from .compare import get_context


def plural(count, word):
    """Return e.g. "1 image" or "5 images"."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def format_tokens(tokens):
    """Return token usage as "8,061 in / 7,987 out tokens", or a note when the API reported no usage (None).

    tokens is a dict (pricing.make_tokens); "in" is the uncached input. Cache writes and reads are added only
    when there are any, e.g. "100 in / 50 out / 2,000 cache read tokens".
    """
    if tokens is None:
        return "token usage not reported"
    parts = [f"{tokens['input']:,} in", f"{tokens['output']:,} out"]
    if tokens['cache_write']:
        parts.append(f"{tokens['cache_write']:,} cache write")
    if tokens['cache_read']:
        parts.append(f"{tokens['cache_read']:,} cache read")
    return " / ".join(parts) + " tokens"


def print_url_report(url_changes):
    """Print removed, added or altered URLs as errors to review (a changed URL is a real error)."""
    if not url_changes:
        print(Fore.GREEN + "\nAll link URLs preserved.")
        return
    print(Fore.RED + Style.BRIGHT + f"\nERROR: {len(url_changes)} URL(s) were removed, added or altered "
          "in the AI-formatted version. Review these:")
    for change in url_changes:
        if change['type'] == 'altered':
            print(Fore.RED + f"  - ALTERED: {change['original']}")
            print(Fore.RED + f"         to: {change['improved']}")
        elif change['type'] == 'removed':
            print(Fore.RED + f"  - REMOVED: {change['original']}")
        else:
            print(Fore.RED + f"  - ADDED:   {change['improved']}")


def print_link_text_edits(comparison_result, verbose=False):
    """Print the expected link text edits (rewritten link labels). Identical edits are collapsed into one entry
    with a count, e.g. "here" -> "" (x3). By default they all go on one line; verbose lists one per line."""
    original_tokens = comparison_result['original_tokens']
    improved_tokens = comparison_result['improved_tokens']
    edits = [m for m in comparison_result['mismatches'] if m['link_text_edit']]
    if not edits:
        return
    # Maps (original words, improved words) -> count, in the order first seen
    counts = {}
    for mismatch in edits:
        original_words = ' '.join(original_tokens[mismatch['original_start']:mismatch['original_end']])
        improved_words = ' '.join(improved_tokens[mismatch['improved_start']:mismatch['improved_end']])
        key = (original_words, improved_words)
        counts[key] = counts.get(key, 0) + 1
    entries = [f'"{original}" -> "{improved}"' + (f" (x{count})" if count > 1 else "")
               for (original, improved), count in counts.items()]
    if verbose:
        print(Fore.CYAN + f"\nLink text edits ({len(edits)}, expected when turning URLs into descriptive links):")
        for entry in entries:
            print(f"  - {entry}")
    else:
        print(Fore.CYAN + f"Link text edits ({len(edits)}, expected): " + Style.RESET_ALL + ", ".join(entries))


def describe_numbers(numbers):
    """Return a list's numbers as "1-4" when they count up by one, else as "1, 2, 5"."""
    if len(numbers) > 1 and all(b == a + 1 for a, b in pairwise(numbers)):
        return f"{numbers[0]}-{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def numbering_entry_text(entry):
    """Return e.g. '2 of 4 items (1-4) at "Open the booking form"'."""
    count = len(entry['numbers'])
    return (f"{entry['changed']} of {plural(count, 'item')} ({describe_numbers(entry['numbers'])}) "
            f"at \"{entry['start_words']}\"")


def print_list_numbering(comparison_result, verbose=False):
    """Print the list numbers that were added, removed or changed (see compare.compare_texts 'numbering').

    Numbers added to or removed from a list are expected formatting and are listed briefly, on one line by
    default and one per line with verbose. A list whose numbers don't count up by one is always a warning;
    its numbers stay in the added words.
    """
    numbering = comparison_result['numbering']
    entries = ([f"added to {numbering_entry_text(e)}" for e in numbering['added']]
               + [f"removed from {numbering_entry_text(e)} of the original" for e in numbering['removed']])
    if entries:
        if verbose:
            print(Fore.CYAN + f"\nList numbering ({len(entries)}, expected when lines become a numbered list "
                  "or a numbered list is renumbered):")
            for entry in entries:
                print(f"  - Numbers {entry}")
        else:
            print(Fore.CYAN + f"List numbering ({len(entries)}, expected): " + Style.RESET_ALL
                  + "; ".join(f"numbers {entry}" for entry in entries))
    for entry in numbering['out_of_order']:
        print(Fore.YELLOW + Style.BRIGHT + f"\nWARNING: a numbered list in the AI-formatted version has its numbers "
              f"out of order ({describe_numbers(entry['numbers'])}) at \"{entry['start_words']}\". "
              "Its new numbers are counted as added words.")


ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')


def visible_len(text):
    """Length of text as displayed, ignoring ANSI color codes."""
    return len(ANSI_ESCAPE_RE.sub('', text))


def print_mismatch_report(comparison_result, detailed=False):
    """Print a detailed report of content mismatches with color coding.

    Expected link text edits and list numbering are left out here; print_link_text_edits and
    print_list_numbering list them separately.
    """
    mismatches = [m for m in comparison_result['mismatches'] if not (m['link_text_edit'] or m['list_numbering'])]
    original_tokens = comparison_result['original_tokens']
    improved_tokens = comparison_result['improved_tokens']
    
    if not mismatches:
        return
    
    # Identical-looking mismatches are printed in full once; repeats get a one-line reference to the first.
    # Maps fingerprint -> number of the first difference printed with it
    printed_fingerprints = {}
    repeated_count = 0
    
    if detailed:
        print(Fore.YELLOW + Style.BRIGHT + f"\nFound {len(mismatches)} content differences (these should be investigated):")
        
        for idx, mismatch in enumerate(mismatches, 1):
            original_context, orig_ctx_start, orig_ctx_end = get_context(
                original_tokens, 
                mismatch['original_start'], 
                mismatch['original_end']
            )
            improved_context, imp_ctx_start, imp_ctx_end = get_context(
                improved_tokens, 
                mismatch['improved_start'], 
                mismatch['improved_end']
            )
            
            # Determine the positions in the context where the mismatch occurs
            # Offset from start of context
            orig_mismatch_offset = mismatch['original_start'] - orig_ctx_start
            orig_mismatch_len = mismatch['original_end'] - mismatch['original_start']
            imp_mismatch_offset = mismatch['improved_start'] - imp_ctx_start
            imp_mismatch_len = mismatch['improved_end'] - mismatch['improved_start']
            
            # Get the actual mismatch tokens
            orig_mismatch_tokens = original_tokens[mismatch['original_start']:mismatch['original_end']]
            imp_mismatch_tokens = improved_tokens[mismatch['improved_start']:mismatch['improved_end']]
            
            # Build strings with highlighted mismatches
            # For original: highlight in RED
            # For improved: highlight in GREEN
            
            # Split context into before, mismatch, after
            original_before = ' '.join(original_tokens[orig_ctx_start:mismatch['original_start']])
            original_mismatch = ' '.join(orig_mismatch_tokens)
            original_after = ' '.join(original_tokens[mismatch['original_end']:orig_ctx_end])
            
            improved_before = ' '.join(improved_tokens[imp_ctx_start:mismatch['improved_start']])
            improved_mismatch = ' '.join(imp_mismatch_tokens)
            improved_after = ' '.join(improved_tokens[mismatch['improved_end']:imp_ctx_end])
            
            # Use [[marker]] to denote the changed content for readability
            if original_mismatch:
                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}{original_mismatch}{Style.RESET_ALL} ]] {original_after}..."
            else:
                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {original_after}..."
            
            if improved_mismatch:
                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}{improved_mismatch}{Style.RESET_ALL} ]] {improved_after}..."
            else:
                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {improved_after}..."
            
            # Ensure lines fit within reasonable length
            max_line_length = 120
            if visible_len(original_line) > max_line_length:
                # Trim the context but keep the mismatch
                i = 0
                while visible_len(original_line) > max_line_length and i < 10:
                    # Trim more from before context
                    if original_before:
                        words = original_before.split()
                        if len(words) > 1:
                            original_before = ' '.join(words[1:])
                            if original_mismatch:
                                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}{original_mismatch}{Style.RESET_ALL} ]] {original_after}..."
                            else:
                                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {original_after}..."
                    # Trim more from after context
                    if visible_len(original_line) > max_line_length and original_after:
                        words = original_after.split()
                        if len(words) > 1:
                            original_after = ' '.join(words[:-1])
                            if original_mismatch:
                                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}{original_mismatch}{Style.RESET_ALL} ]] {original_after}..."
                            else:
                                original_line = f"ORIGINAL: ...{original_before} [[ {Fore.RED + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {original_after}..."
                    i += 1
                
            if visible_len(improved_line) > max_line_length:
                i = 0
                while visible_len(improved_line) > max_line_length and i < 10:
                    if improved_before:
                        words = improved_before.split()
                        if len(words) > 1:
                            improved_before = ' '.join(words[1:])
                            if improved_mismatch:
                                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}{improved_mismatch}{Style.RESET_ALL} ]] {improved_after}..."
                            else:
                                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {improved_after}..."
                    if visible_len(improved_line) > max_line_length and improved_after:
                        words = improved_after.split()
                        if len(words) > 1:
                            improved_after = ' '.join(words[:-1])
                            if improved_mismatch:
                                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}{improved_mismatch}{Style.RESET_ALL} ]] {improved_after}..."
                            else:
                                improved_line = f"IMPROVED: ...{improved_before} [[ {Fore.GREEN + Style.BRIGHT}[[NONE]]{Style.RESET_ALL} ]] {improved_after}..."
                    i += 1
            
            # Create a unique fingerprint for this mismatch to avoid duplication
            # Use the mismatch context as the fingerprint
            mismatch_header = f"Content Difference #{idx} ({mismatch['type']}):"
            
            # Print repeats of an already-printed mismatch as a short reference instead of in full
            if original_line and improved_line:
                
                # Create a fingerprint for this mismatch
                fingerprint = f"{original_line}|{improved_line}"
                
                if fingerprint in printed_fingerprints:
                    repeated_count += 1
                    print(Fore.CYAN + f"\n{mismatch_header} same text as Content Difference "
                          f"#{printed_fingerprints[fingerprint]} (shown above)")
                else:
                    printed_fingerprints[fingerprint] = idx
                    
                    header_emoji = "⚠️ "  # Default warning
                    
                    # Pick the header symbol from the difference type
                    if "(delete)" in mismatch_header:
                        header_emoji = "❌"
                    elif "(insert)" in mismatch_header:
                        header_emoji = "➕"
                    elif "(replace)" in mismatch_header:
                        header_emoji = "🔄"
                    
                    # Print Header (Cyan)
                    print(Fore.CYAN + Style.BRIGHT + f"\n{header_emoji} {mismatch_header}")
                    
                    # Print lines (they contain their own colors now)
                    # Add a small indent for readability
                    print(f"   {original_line}")
                    print(f"   {improved_line}")
        if repeated_count:
            print(Fore.YELLOW + f"\n{repeated_count} of {len(mismatches)} differences repeat an earlier one "
                  "and were listed by reference only.")
    else:
        print("Run with --detailed-comparison to see content differences.")


def print_added_alt_text_report(generated, warnings):
    """Print the alt text the model added to the document (for the user to check), and images without alt text."""
    if generated:
        print(Fore.YELLOW + Style.BRIGHT + f"\n===== ADDED TEXT - PLEASE CHECK: {len(generated)} alt text "
              "description(s) were written by the model and added to the document =====")
        for path, alt_text, _heading in generated:
            print(Fore.YELLOW + Style.BRIGHT + f"\n  Image:   {path}")
            print(f"  Alt:     {alt_text}")
        print(Fore.YELLOW + Style.BRIGHT + "\nReview each description for accuracy before publishing.")
    if warnings:
        print(Fore.RED + Style.BRIGHT + f"\nWARNING: {len(warnings)} image(s) did not get alt text:")
        for image_ref, reason in warnings:
            print(Fore.RED + f"  - {image_ref}: {reason}")


def print_preserved_summary(comparison_result):
    """Print one line saying what was preserved (link URLs, original words), if anything was.

    Used in the default (not verbose) output in place of the separate success lines of print_url_report and
    print_missing_and_added_words; the problems themselves are printed by those functions.
    """
    preserved = []
    if not comparison_result['url_changes']:
        preserved.append("All link URLs preserved.")
    if not comparison_result['missing_words']:
        preserved.append("All original words preserved.")
    if preserved:
        print(Fore.GREEN + "\n" + " ".join(preserved))


def print_missing_and_added_words(comparison_result, verbose):
    """Print the words missing from the improved version (always). With verbose, also print that all words were
    kept (when they were) and the added words."""
    # Missing/added words count only differences that are not expected link text edits or list numbers
    if comparison_result['missing_words']:
        print(Fore.RED + Style.BRIGHT + "\nWARNING: Words missing in the AI-formatted version:")
        for word, count in comparison_result['missing_words'].items():
            print(Fore.RED + f"  - '{word}' appears {count} fewer times")
        print(Fore.RED + "\nThis should not happen! Please review the output carefully.")
    elif verbose:
        print(Fore.GREEN + "\nAll original words preserved. Content preservation verified!")

    if comparison_result['added_words'] and verbose:
        print("\nNOTE: Words added in the AI-formatted version (outside expected link text edits and list "
              "numbering):")
        for word, count in comparison_result['added_words'].items():
            print(f"  - '{word}' appears {count} more times")
        print("\nThis may be due to formatting. Please review the output carefully.")
