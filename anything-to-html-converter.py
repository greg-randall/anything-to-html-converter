import os
import re
import argparse
from collections import Counter
import openai
import difflib
import pypandoc
import time
from mistralai import Mistral
import httpx
import logging
import subprocess


logger = logging.getLogger('my-logger')
logger.propagate = False



def convert_docx_to_markdown(docx_path, output_path):
    """Convert Word document to markdown using pypandoc."""
    try:
        # Set extra arguments for pandoc
        extra_args = ["--wrap=none", "--extract-media=./media"]

        # Use pypandoc to convert docx to markdown.
        # The file will be saved directly to output_path.
        pypandoc.convert_file(docx_path, 'markdown', outputfile=output_path, extra_args=extra_args)
        
        print(f"Successfully converted {docx_path} to {output_path}")
        
        # Read the markdown content
        with open(output_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            
        # Unwrap paragraphs
        unwrapped_content = unwrap_markdown_paragraphs(markdown_content)
        
        # Convert pandoc markdown to standard markdown
        standard_markdown = convert_pandoc_markdown_to_standard(unwrapped_content)
        
        # Write back the processed content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(standard_markdown)
        
        return standard_markdown
    except Exception as e:
        print(f"Error: {e}")
        return None

def unwrap_markdown_paragraphs(markdown_text):
    """
    Remove line breaks within paragraphs in markdown text.
    Preserves line breaks between paragraphs and list items.
    """
    # Split text by double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', markdown_text)
    
    processed_paragraphs = []
    for para in paragraphs:
        # Skip code blocks, headings, and lists
        if (re.match(r'^\s*```', para) or 
            re.match(r'^\s*#', para) or 
            re.match(r'^\s*[*+-]\s', para) or 
            re.match(r'^\s*\d+\.\s', para)):
            processed_paragraphs.append(para)
            continue
        
        # Replace single newlines with spaces within regular paragraphs
        unwrapped = re.sub(r'\n(?!\n)', ' ', para)
        processed_paragraphs.append(unwrapped)
    
    # Join paragraphs back with double newlines
    return '\n\n'.join(processed_paragraphs)

def convert_pandoc_markdown_to_standard(markdown_text):
    """
    Convert Pandoc-specific markdown to more standard markdown that GPT can understand.
    """
    # Replace pandoc's {.underline} with standard markdown underline syntax
    markdown_text = re.sub(r'\{\s*\.underline\s*\}', '', markdown_text)
    
    # Replace [text]{.underline} with **text**
    markdown_text = re.sub(r'\[(.*?)\]\{\s*\.underline\s*\}', r'**\1**', markdown_text)
    
    # Clean up heading brackets - pattern **[Heading]** or *[Heading]**
    markdown_text = re.sub(r'\*+\s*\[(.*?)\]\s*\*+', r'**\1**', markdown_text)
    
    # Clean up heading brackets in potential heading lines
    pattern = r'^(\s*)\[(.*?)\]:(.*?)$'
    replacement = r'\1\2:\3'
    markdown_text = re.sub(pattern, replacement, markdown_text, flags=re.MULTILINE)
    
    # Remove other pandoc-specific class attributes
    markdown_text = re.sub(r'\{\s*\.[a-zA-Z0-9_-]+\s*\}', '', markdown_text)
    
    return markdown_text

def tokenize_text(text):
    """
    Tokenize text by extracting only words and numbers, ignoring formatting markers and link URLs.
    """
    # First, handle markdown links - extract only the link text, ignore the URLs
    text = re.sub(r'\[(.*?)\]\(.*?\)', r' \1 ', text)
    
    # Handle HTML links
    text = re.sub(r'<a\s+href=[^>]*>(.*?)</a>', r' \1 ', text)
    
    # Extract only alphanumeric tokens (words and numbers)
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    
    # Filter out common formatting markers
    formatting_markers = ['underline', 'bold', 'italic', 'strikethrough', 'highlight', 'br']
    tokens = [token for token in tokens if token.lower() not in formatting_markers]
    
    return tokens

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
                # Take the more specific type if available
                if current['type'] != 'replace' and previous['type'] == 'replace':
                    previous['type'] = current['type']
            else:
                merged_mismatches.append(current)
        
        return merged_mismatches
    
    return mismatches

def detect_url_to_link_transformation(original_segment, improved_segment):
    """
    Detect if a raw URL in the original was transformed into a text link in the improved version.
    
    Args:
        original_segment (list): List of tokens from the original text
        improved_segment (list): List of tokens from the improved text
        
    Returns:
        bool: True if this appears to be a URL-to-link transformation, False otherwise
    """
    # Join the original tokens to check for URL patterns
    original_text = ' '.join(original_segment)
    
    # Simple URL pattern detection (can be expanded for more complex patterns)
    url_pattern = r'https?|www\.|\.com|\.edu|\.org|\.net|\.gov'
    
    # Check if original contains URL-like patterns
    contains_url = bool(re.search(url_pattern, original_text, re.IGNORECASE))
    
    # If the original contains URL-like patterns and the improved version is different,
    # this might be a URL-to-link transformation
    return contains_url and original_segment != improved_segment

def show_mismatch_context(original_tokens, improved_tokens, mismatch):
    """Show context around a mismatch, always showing both original and improved versions."""
    result = []
    
    context_size = 10  # Increased context size for better visibility
    
    # For original version
    orig_mid_point = (mismatch['original_start'] + mismatch['original_end']) // 2
    orig_start = max(0, orig_mid_point - context_size)
    orig_end = min(len(original_tokens), orig_mid_point + context_size)
    
    # For improved version
    imp_mid_point = (mismatch['improved_start'] + mismatch['improved_end']) // 2
    imp_start = max(0, imp_mid_point - context_size)
    imp_end = min(len(improved_tokens), imp_mid_point + context_size)
    
    # Original context and content
    orig_prefix = original_tokens[orig_start:mismatch['original_start']]
    orig_content = original_tokens[mismatch['original_start']:mismatch['original_end']]
    orig_suffix = original_tokens[mismatch['original_end']:orig_end]
    
    # Improved context and content
    imp_prefix = improved_tokens[imp_start:mismatch['improved_start']]
    imp_content = improved_tokens[mismatch['improved_start']:mismatch['improved_end']]
    imp_suffix = improved_tokens[mismatch['improved_end']:imp_end]
    
    # Check if this might be a URL-to-link transformation
    is_url_transformation = detect_url_to_link_transformation(orig_content, imp_content)
    
    # Add a note if this appears to be a URL transformation
    if is_url_transformation:
        result.append("NOTE: This appears to be a URL converted to a text link (expected formatting improvement)")
    
    # Format original line
    prefix_str = ' '.join(orig_prefix)
    content_str = ' '.join(orig_content) if orig_content else "[[NONE]]"
    suffix_str = ' '.join(orig_suffix)
    result.append(f"ORIGINAL: ...{prefix_str} [[ {content_str} ]] {suffix_str}...")
    
    # Format improved line
    prefix_str = ' '.join(imp_prefix)
    content_str = ' '.join(imp_content) if imp_content else "[[NONE]]"
    suffix_str = ' '.join(imp_suffix)
    result.append(f"IMPROVED: ...{prefix_str} [[ {content_str} ]] {suffix_str}...")
    
    return '\n'.join(result)

def compare_texts(original, improved):
    """Compare original and improved text with extreme strictness to catch any content changes."""
    # Tokenize both texts to extract only words and numbers
    original_tokens = tokenize_text(original)
    improved_tokens = tokenize_text(improved)
    
    # Find mismatches (case-insensitive comparison of alphanumeric tokens only)
    mismatches = find_mismatches(original_tokens, improved_tokens)
    
    # Create a hash set to track mismatch fingerprints to avoid duplication
    reported_fingerprints = set()
    
    # Prepare mismatch report
    mismatch_report = []
    
    if mismatches:
        mismatch_report.append(f"Found {len(mismatches)} content differences (these should be investigated):")
        for i, mismatch in enumerate(mismatches, 1):
            # Create a fingerprint for this mismatch
            orig_content = ' '.join(original_tokens[mismatch['original_start']:mismatch['original_end']]) if mismatch['original_start'] < mismatch['original_end'] else "[[NONE]]"
            imp_content = ' '.join(improved_tokens[mismatch['improved_start']:mismatch['improved_end']]) if mismatch['improved_start'] < mismatch['improved_end'] else "[[NONE]]"
            fingerprint = f"{orig_content}|{imp_content}"
            
            # Skip if we've already reported this mismatch
            if fingerprint in reported_fingerprints:
                continue
                
            reported_fingerprints.add(fingerprint)
            
            mismatch_report.append(f"\nContent Difference #{i} ({mismatch['type']}):")
            mismatch_report.append(show_mismatch_context(original_tokens, improved_tokens, mismatch))
    
    # Perform word count comparison for additional verification
    original_words = [token.lower() for token in original_tokens]
    improved_words = [token.lower() for token in improved_tokens]
    
    original_counter = Counter(original_words)
    improved_counter = Counter(improved_words)
    
    missing_words = {}
    added_words = {}
    
    # Check for missing words
    for word, count in original_counter.items():
        if word not in improved_counter or improved_counter[word] < count:
            missing_count = count - improved_counter.get(word, 0)
            missing_words[word] = missing_count
    
    # Check for added words
    for word, count in improved_counter.items():
        if word not in original_counter or original_counter[word] < count:
            added_count = count - original_counter.get(word, 0)
            added_words[word] = added_count
    
    return {
        'mismatches': mismatches,
        'missing_words': missing_words,
        'added_words': added_words,
        'mismatch_report': '\n'.join(mismatch_report) if mismatch_report else "No content differences found."
    }

def post_process_improved_markdown(markdown_text):
    """
    Perform post-processing on the improved markdown to fix common issues.
    """
    # Remove square brackets that completely wrap heading text
    # Pattern: Find headings (## ) followed by text in square brackets
    markdown_text = re.sub(r'(#+\s+)\[+([^\]]+)\]+(\s*(?:\n|$))', r'\1\2\3', markdown_text)
    
    return markdown_text


def process_document_with_ocr(api_key, document_path, max_retries=3, retry_delay=2, debug=False):
    """
    Process a local document with Mistral's OCR service with improved error handling
    
    Args:
        api_key (str): Mistral API key
        document_path (str): Path to the local document file
        max_retries (int): Maximum number of retry attempts for API calls
        retry_delay (int): Delay between retries in seconds
        debug (bool): Whether to show detailed logging output
        
    Returns:
        The OCR response from Mistral or None if processing failed
    """

    # Validate the file exists
    if not os.path.exists(document_path):
        logger.error(f"File not found: {document_path}")
        return None
    
    # Check file size (Mistral has a 50MB limit)
    file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
    if file_size_mb > 50:
        logger.error(f"File size ({file_size_mb:.2f}MB) exceeds Mistral's 50MB limit")
        return None
    
    # Initialize the Mistral client
    client = Mistral(
        api_key=api_key,
    )
    
    # Get the filename from the path
    filename = os.path.basename(document_path)
    logger.info(f"Processing file: {filename} ({file_size_mb:.2f}MB)")
    
    # Upload the file with retry logic
    uploaded_file = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Uploading file: {filename} (Attempt {attempt+1}/{max_retries})")
            with open(document_path, "rb") as file_handle:
                uploaded_file = client.files.upload(
                    file={
                        "file_name": filename,
                        "content": file_handle,
                    },
                    purpose="ocr"
                )
            logger.info(f"File uploaded successfully with ID: {uploaded_file.id}")
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retry attempts reached. Upload failed.")
                return None
    
    if not uploaded_file:
        return None
    
    # Get the signed URL with retry logic
    signed_url = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Getting signed URL (Attempt {attempt+1}/{max_retries})")
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
            logger.info("Signed URL obtained successfully")
            break
        except Exception as e:
            logger.error(f"Getting signed URL attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retry attempts reached. Failed to get signed URL.")
                return None
    
    if not signed_url:
        return None
    
    # Process with OCR with retry logic
    ocr_response = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Processing document with OCR (Attempt {attempt+1}/{max_retries})")
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
            logger.info("OCR processing completed successfully")
            break
        except Exception as e:
            logger.error(f"OCR processing attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retry attempts reached. OCR processing failed.")
                return None
    
    if ocr_response:
        logger.info("Document processing complete")
        return ocr_response
    else:
        return None


def extract_markdown_from_ocr(ocr_response, output_path):
    """Extract markdown content from OCR response and save to file."""
    try:
        # Check if the result has a 'pages' attribute (structured OCR response)
        if hasattr(ocr_response, 'pages'):
            # Extract text from each page and join with page breaks
            extracted_text = ""
            for page in ocr_response.pages:
                if hasattr(page, 'markdown'):
                    extracted_text += page.markdown + "\n\n"
                elif hasattr(page, 'text'):
                    extracted_text += page.text + "\n\n"
        else:
            # Fallback to using whatever text attribute is available
            extracted_text = ocr_response.text if hasattr(ocr_response, 'text') else str(ocr_response)
        
        # Write the extracted text to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        print(f"OCR extracted text saved to: {output_path}")
        return extracted_text
    except Exception as e:
        print(f"Error extracting OCR text: {str(e)}")
        return None


def convert_markdown_to_raw_html(markdown_path, html_path):
    """Convert markdown to raw HTML using the markdown library."""
    try:
        import markdown
        
        # Read the markdown file
        with open(markdown_path, 'r', encoding='utf-8') as md_file:
            md_content = md_file.read()
        
        # Convert to HTML - this produces just the HTML content, no document structure
        html_content = markdown.markdown(md_content)
        
        # Write to the output file
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)
        
        print(f"Successfully converted {markdown_path} to raw HTML at {html_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def improve_markdown_with_gpt(api_key, markdown_content):
    """Improve markdown formatting using GPT-4o-mini."""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = """
Read the Markdown below and **only** fix formatting—never touch any words or typos.

1. Normalize headings so they descend logically (`#`, `##`, `###`, …).  
2. Turn raw URLs into inline links using the same text.  
3. Format anything that looks like a list into a proper list.  
4. Convert messy rows (e.g. hyphens, tabs, or multiple spaces separating values) into tables.  
5. Make the whole thing beautiful and consistent.  
6. **Do not** wrap your answer in code fences.

**CRITICAL:** Preserve every character, typo, and duplicate exactly as is.

---

Here's the content to reformat:

{markdown}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that improves markdown formatting without changing a single word of the original content."},
                {"role": "user", "content": prompt.format(markdown=markdown_content)}
            ]
        )
        
        improved_markdown = response.choices[0].message.content
        
        # Post-process the improved markdown to fix any remaining issues
        improved_markdown = post_process_improved_markdown(improved_markdown)
        
        return improved_markdown
    except Exception as e:
        print(f"Error using OpenAI API: {e}")
        return None

def print_mismatch_report(comparison_result, detailed=True):
    """
    Print a mismatch report with deduplication.
    This function ensures we only print each unique mismatch once.
    """
    if not comparison_result['mismatches']:
        print("No content differences detected. All words are preserved exactly in the final version.")
        return
    
    # Create a set to track printed mismatch fingerprints
    printed_fingerprints = set()
    
    print(f"\nFound {len(comparison_result['mismatches'])} content differences (these should be investigated):")
    
    # Extract the mismatch details from the report
    if detailed:
        report_lines = comparison_result['mismatch_report'].split("\n")
        i = 0
        while i < len(report_lines):
            line = report_lines[i]
            
            if line.startswith("Content Difference #"):
                # Extract the mismatch type
                mismatch_header = line
                i += 1
                
                # Check if the next line is a note about URL transformation
                is_url_transformation = False
                if i < len(report_lines) and report_lines[i].startswith("NOTE: This appears to be a URL"):
                    is_url_transformation = True
                    url_note = report_lines[i]
                    i += 1
                
                # The next two lines should be the original and improved
                original_line = report_lines[i] if i < len(report_lines) else ""
                i += 1
                improved_line = report_lines[i] if i < len(report_lines) else ""
                i += 1
                
                # Create a fingerprint for this mismatch
                fingerprint = f"{original_line}|{improved_line}"
                
                # Only print this mismatch if we haven't printed it before
                if fingerprint not in printed_fingerprints:
                    printed_fingerprints.add(fingerprint)
                    
                    # Modify the header if this is a URL transformation
                    if is_url_transformation:
                        mismatch_header = mismatch_header.replace("Content Difference", "URL-to-Link Transformation")
                    
                    print(f"\n{mismatch_header}")
                    if is_url_transformation:
                        print(url_note)
                    print(original_line)
                    print(improved_line)
            else:
                i += 1
    else:
        print("Run with --detailed-comparison to see content differences.")

def main():
    parser = argparse.ArgumentParser(description='Convert Word to Markdown and improve formatting')
    parser.add_argument('input_file', help='Path to input Word document')
    parser.add_argument('--output', help='Path to output Markdown file (default: input filename with .md extension)')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--mistral-api-key', help='Mistral API key for OCR')
    parser.add_argument('--show-diff', action='store_true', help='Show diff between original and improved markdown')
    # By default, detailed comparison is enabled. To disable it, use --no-detailed-comparison.
    parser.add_argument('--no-detailed-comparison', dest='detailed_comparison', action='store_false', help='Disable detailed token-by-token comparison')
    parser.set_defaults(detailed_comparison=True)
    # New flag: by default, only the improved HTML file will be saved.
    # Use --keep-markdown to keep the intermediate markdown files.
    parser.add_argument('--keep-markdown', action='store_true', help='Keep the intermediate markdown files (default: do not keep them)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set default output path for markdown if not provided
    if not args.output:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}.md"
    
    # Get OpenAI API key
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    # Get Mistral API key
    mistral_api_key = args.mistral_api_key or os.environ.get("MISTRAL_API_KEY")
    
    input_file = args.input_file
    
    # Always try to use Mistral OCR first, fallback to pandoc if necessary
    if mistral_api_key:
        print(f"Converting {input_file} using Mistral OCR...")
        ocr_response = process_document_with_ocr(mistral_api_key, input_file, debug=args.debug)
        if ocr_response:
            original_markdown = extract_markdown_from_ocr(ocr_response, args.output)
            if not original_markdown:
                print("OCR extraction failed. Falling back to pandoc...")
                original_markdown = convert_docx_to_markdown(input_file, args.output)
        else:
            print("OCR processing failed. Falling back to pandoc...")
            original_markdown = convert_docx_to_markdown(input_file, args.output)
    else:
        print(f"Mistral API key not provided. Converting {input_file} using pandoc...")
        original_markdown = convert_docx_to_markdown(input_file, args.output)
    
    if original_markdown is None:
        print("Conversion failed. Exiting.")
        return

    # If we are not keeping markdown files, remove the original markdown file after reading its content
    if not args.keep_markdown:
        try:
            if os.path.exists(args.output):
                os.remove(args.output)
        except Exception as e:
            print(f"Warning: could not remove temporary markdown file: {e}")
    
    # Check if API key is provided
    if not openai_api_key:
        print("OpenAI API key is required for GPT improvement.")
        print("Either pass it with --api-key or set the OPENAI_API_KEY environment variable.")
        return

    # Improve markdown with GPT
    print("Improving markdown with GPT-4o-mini...")
    improved_markdown = improve_markdown_with_gpt(openai_api_key, original_markdown)
    if improved_markdown is None:
        print("GPT improvement failed. Using original markdown.")
        final_markdown = original_markdown
    else:
        final_markdown = improved_markdown
        # Optionally, if you want to keep the improved markdown file,
        # write it to disk only if --keep-markdown is provided.
        if args.keep_markdown:
            improved_output = f"{os.path.splitext(args.output)[0]}_improved.md"
            with open(improved_output, 'w', encoding='utf-8') as f:
                f.write(improved_markdown)
            print(f"Improved markdown saved to {improved_output}")

    # Always convert the final markdown to HTML
    html_output = f"{os.path.splitext(args.input_file)[0]}_improved.html"
    try:
        import pypandoc
        extra_args = ["--standalone", "--metadata", "title=Document"]
        html_content = pypandoc.convert_text(final_markdown, 'html', format='markdown', extra_args=extra_args)
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML conversion complete. HTML saved to {html_output}")
    except Exception as e:
        print(f"Error converting markdown to HTML: {e}")
        return

    # Compare texts to ensure no content was lost
    comparison_result = compare_texts(original_markdown, final_markdown)

    if comparison_result['missing_words']:
        print("\nWARNING: Words missing in the improved version:")
        for word, count in comparison_result['missing_words'].items():
            print(f"  - '{word}' appears {count} fewer times")
        print("\nThis should not happen! Please review the output carefully.")
    else:
        print("\nAll original words preserved. Content preservation verified!")
    
    if comparison_result.get('added_words'):
        print("\nWARNING: Words added in the improved version:")
        for word, count in comparison_result['added_words'].items():
            print(f"  - '{word}' appears {count} more times")
        print("\nThis should not happen! Please review the output carefully.")

    # Use the dedicated function to print mismatch report without duplication
    if args.detailed_comparison and comparison_result['mismatches']:
        print_mismatch_report(comparison_result, detailed=True)
    elif comparison_result['mismatches']:
        print(f"\nFound {len(comparison_result['mismatches'])} content differences.")
        print("To see details, run without --no-detailed-comparison")
        print("\nIMPORTANT: Content differences should not occur! This needs investigation.")
    
    if args.show_diff:
        print("\n===== UNIFIED LINE DIFF =====")
        diff = difflib.unified_diff(
            original_markdown.splitlines(),
            final_markdown.splitlines(),
            lineterm='',
            n=3
        )
        print('\n'.join(diff))

if __name__ == "__main__":
    main()
