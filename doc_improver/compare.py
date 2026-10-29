"""Comparing the original and improved text word by word."""
import difflib
from collections import Counter

from .links import analyze_markdown, compare_urls, is_link_text_edit


def find_mismatches(original_tokens, improved_tokens):
    """Find mismatches between two token sequences using diff, case-insensitive."""
    # Create lowercase versions of tokens for comparison
    original_lower = [t.lower() for t in original_tokens]
    improved_lower = [t.lower() for t in improved_tokens]
    
    matcher = difflib.SequenceMatcher(None, original_lower, improved_lower)
    mismatches = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # This is a mismatch (replace, delete, or insert)
            mismatches.append({
                'type': tag,
                'original_start': i1,
                'original_end': i2,
                'improved_start': j1,
                'improved_end': j2,
                # The changed sub-ranges; a merged region can also contain unchanged tokens between them
                'parts': [(i1, i2, j1, j2)],
            })
    
    # Merge adjacent or overlapping mismatches
    if mismatches:
        merged_mismatches = [mismatches[0]]
        for current in mismatches[1:]:
            previous = merged_mismatches[-1]
            
            # Define a threshold for considering mismatches as adjacent
            # Adjust this value based on your specific needs
            adjacency_threshold = 5
            
            # Check if current mismatch is adjacent to or overlaps with previous mismatch
            if (current['original_start'] <= previous['original_end'] + adjacency_threshold and 
                current['improved_start'] <= previous['improved_end'] + adjacency_threshold):
                # Merge the mismatches
                previous['original_end'] = max(previous['original_end'], current['original_end'])
                previous['improved_end'] = max(previous['improved_end'], current['improved_end'])
                previous['parts'].extend(current['parts'])
            else:
                merged_mismatches.append(current)

        # Label each (possibly merged) region by what it actually contains
        for mismatch in merged_mismatches:
            has_original = mismatch['original_end'] > mismatch['original_start']
            has_improved = mismatch['improved_end'] > mismatch['improved_start']
            if has_original and has_improved:
                mismatch['type'] = 'replace'
            elif has_original:
                mismatch['type'] = 'delete'
            else:
                mismatch['type'] = 'insert'

        return merged_mismatches
    
    return mismatches


def get_context(tokens, start, end, context_window=20):
    """Get context around a specific token range."""
    context_start = max(0, start - context_window)
    context_end = min(len(tokens), end + context_window)
    return ' '.join(tokens[context_start:context_end]), context_start, context_end


def list_numbers_in(mismatch, original_analysis, improved_analysis):
    """Return (original positions, improved positions) of the changed tokens of mismatch that are list item
    numbers. On the improved side only numbers of a list that counts up by one are returned; numbers of a list
    out of order (e.g. 1, 2, 5) stay ordinary added words."""
    original_items = original_analysis['list_item']
    improved_items = improved_analysis['list_item']
    improved_lists = improved_analysis['ordered_lists']
    original_positions = []
    improved_positions = []
    for i1, i2, j1, j2 in mismatch['parts']:
        original_positions.extend(i for i in range(i1, i2) if original_items[i] is not None)
        improved_positions.extend(j for j in range(j1, j2) if improved_items[j] is not None
                                  and improved_lists[improved_items[j]]['consecutive'])
    return original_positions, improved_positions


def numbering_entries(ordered_lists, tokens, positions):
    """Return one entry per list that has an item number at one of positions: {numbers, changed (how many of
    its numbers are at positions), start_words (up to 5 words of its first item)}."""
    entries = []
    for found in ordered_lists:
        changed = sum(1 for p in found['positions'] if p in positions)
        if changed:
            first = found['positions'][0]
            end = found['positions'][1] if len(found['positions']) > 1 else len(tokens)
            entries.append({'numbers': found['numbers'], 'changed': changed,
                            'start_words': ' '.join(tokens[first + 1:min(end, first + 6)])})
    return entries


def compare_texts(original_text, improved_text):
    """
    Compare original and improved text to check for missing or added words and changed URLs.
    Returns a dictionary with comparison results.

    Each mismatch gets a 'link_text_edit' flag (an expected link label rewrite, see is_link_text_edit) and a
    'list_numbering' flag (every changed token is a list item number: numbering added, removed or changed).
    List item numbers are left out of missing_words / added_words wherever they appear in a difference, except
    the numbers of an improved list that doesn't count up by one; those are in 'numbering' 'out_of_order'.
    missing_words / added_words count only the other words of differences that are NOT link text edits.
    """
    original_analysis = analyze_markdown(original_text)
    improved_analysis = analyze_markdown(improved_text)
    original_tokens = original_analysis['tokens']
    improved_tokens = improved_analysis['tokens']

    # Find token-level mismatches
    mismatches = find_mismatches(original_tokens, improved_tokens)

    # Bare URLs (shown as themselves) in the original that are link targets in the improved version.
    # The words directly before such a URL may be rewritten into the new link's label.
    improved_link_urls = {u['url'] for u in improved_analysis['urls'] if u['kind'] == 'link'}
    converted_url_positions = {u['pos'] for u in original_analysis['urls']
                               if u['kind'] == 'bare' and u['url'] in improved_link_urls}

    # Classify mismatches and collect the token positions explained as link text edits
    explained_original = set()
    explained_improved = set()
    for mismatch in mismatches:
        mismatch['link_text_edit'] = is_link_text_edit(mismatch, original_analysis, improved_analysis,
                                                       converted_url_positions)
        if mismatch['link_text_edit']:
            for i1, i2, j1, j2 in mismatch['parts']:
                explained_original.update(range(i1, i2))
                explained_improved.update(range(j1, j2))

    # List item numbers added, removed or changed (e.g. plain lines turned into a numbered list)
    numbers_original = set()
    numbers_improved = set()
    changed_improved = set()
    for mismatch in mismatches:
        mismatch['list_numbering'] = False
        if mismatch['link_text_edit']:
            continue
        original_positions, improved_positions = list_numbers_in(mismatch, original_analysis, improved_analysis)
        numbers_original.update(original_positions)
        numbers_improved.update(improved_positions)
        changed_count = 0
        for i1, i2, j1, j2 in mismatch['parts']:
            changed_count += (i2 - i1) + (j2 - j1)
            changed_improved.update(range(j1, j2))
        mismatch['list_numbering'] = len(original_positions) + len(improved_positions) == changed_count
    explained_original |= numbers_original
    explained_improved |= numbers_improved
    out_of_order = [found for found in improved_analysis['ordered_lists'] if not found['consecutive']]
    numbering = {
        'added': numbering_entries(improved_analysis['ordered_lists'], improved_tokens, numbers_improved),
        'removed': numbering_entries(original_analysis['ordered_lists'], original_tokens, numbers_original),
        'out_of_order': numbering_entries(out_of_order, improved_tokens, changed_improved),
    }

    # Count occurrences of each word (case-insensitive), leaving out expected link text edits and list numbers
    original_counts = Counter(token.lower() for i, token in enumerate(original_tokens) if i not in explained_original)
    improved_counts = Counter(token.lower() for i, token in enumerate(improved_tokens) if i not in explained_improved)

    # Words that appear less frequently in improved version (missing words)
    missing_words = dict(original_counts - improved_counts)

    # Words that appear more frequently in improved version (added words)
    added_words = dict(improved_counts - original_counts)

    return {
        'original_count': len(original_tokens),
        'improved_count': len(improved_tokens),
        'missing_words': missing_words,
        'added_words': added_words,
        'mismatches': mismatches,
        'url_changes': compare_urls(original_analysis['urls'], improved_analysis['urls']),
        'numbering': numbering,
        'original_tokens': original_tokens,
        'improved_tokens': improved_tokens
    }
