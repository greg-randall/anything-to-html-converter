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


# Add these imports for LibreOffice conversion
try:
    import uno
    from com.sun.star.beans import PropertyValue
    UNO_AVAILABLE = True
except ImportError:
    UNO_AVAILABLE = False



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
    formatting_markers = ['underline', 'bold', 'italic', 'strikethrough', 'highlight']
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
    
    return mismatches

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
    
    # Prepare mismatch report
    mismatch_report = []
    
    if mismatches:
        mismatch_report.append(f"Found {len(mismatches)} content differences (these should be investigated):")
        for i, mismatch in enumerate(mismatches, 1):
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


def ensure_libreoffice_running(port=2002):
    """Ensure LibreOffice is running in listening mode."""
    if not UNO_AVAILABLE:
        logger.error("python-uno is not installed. Cannot use LibreOffice conversion.")
        return False
        
    try:
        # Try to connect to LibreOffice
        local_context = uno.getComponentContext()
        resolver = local_context.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_context)
        resolver.resolve(f"uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext")
        logger.info("LibreOffice is already running in listening mode")
        return True
    except Exception:
        # Start LibreOffice in listening mode
        logger.info("Starting LibreOffice in listening mode")
        try:
            # Use subprocess.Popen to start LibreOffice in background
            subprocess.Popen([
                'soffice',
                '--accept=socket,host=localhost,port=2002;urp;',
                '--headless',
                '--nocrashreport',
                '--nodefault',
                '--nofirststartwizard',
                '--nolockcheck',
                '--nologo',
                '--norestore'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Give it a moment to start
            time.sleep(3)
            return True
        except Exception as e:
            logger.error(f"Failed to start LibreOffice: {e}")
            return False

def convert_to_pdf_with_libreoffice(input_file, output_dir=None):
    """
    Convert various document formats to PDF using LibreOffice.
    Supported formats include: doc, docx, odt, rtf, txt, etc.
    
    Args:
        input_file: Path to the input document
        output_dir: Directory to save the PDF (defaults to same directory as input)
        
    Returns:
        Path to the generated PDF or None if conversion failed
    """
    if not UNO_AVAILABLE:
        logger.error("python-uno is not installed. Cannot use LibreOffice conversion.")
        return None
        
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return None
        
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.pdf")
    
    # Ensure LibreOffice is running
    if not ensure_libreoffice_running():
        logger.error("Could not start LibreOffice. Is it installed?")
        return None
    
    try:
        # Get the component context
        local_context = uno.getComponentContext()
        
        # Create desktop service
        resolver = local_context.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local_context)
        
        # Connect to running LibreOffice instance
        context = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
        desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
        
        # Convert paths to URLs
        url_in = uno.systemPathToFileUrl(os.path.abspath(input_file))
        url_out = uno.systemPathToFileUrl(os.path.abspath(output_file))
        
        # Load the document
        logger.info(f"Loading document: {input_file}")
        doc = desktop.loadComponentFromURL(url_in, "_blank", 0, ())
        
        # Set properties for PDF export
        properties = []
        
        # PDF export properties
        p = PropertyValue()
        p.Name = "FilterName"
        p.Value = "writer_pdf_Export"
        properties.append(p)
        
        # Save the document as PDF
        logger.info(f"Converting to PDF: {output_file}")
        doc.storeToURL(url_out, tuple(properties))
        doc.close(True)
        
        if os.path.exists(output_file):
            logger.info(f"Successfully converted to PDF: {output_file}")
            return output_file
        else:
            logger.error("PDF conversion failed: Output file not created")
            return None
            
    except Exception as e:
        logger.error(f"Error converting to PDF: {str(e)}")
        return None

def convert_to_pdf_with_unoconv(input_file, output_dir=None):
    """
    Fallback method to convert documents to PDF using unoconv command-line tool.
    
    Args:
        input_file: Path to the input document
        output_dir: Directory to save the PDF (defaults to same directory as input)
        
    Returns:
        Path to the generated PDF or None if conversion failed
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return None
        
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.pdf")
    
    try:
        logger.info(f"Converting to PDF using unoconv: {input_file}")
        subprocess.run(['unoconv', '-f', 'pdf', '-o', output_dir, input_file], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if os.path.exists(output_file):
            logger.info(f"Successfully converted to PDF: {output_file}")
            return output_file
        else:
            logger.error("PDF conversion failed: Output file not created")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"unoconv conversion failed: {e}")
        return None
    except FileNotFoundError:
        logger.error("unoconv not found. Please install it with: apt-get install unoconv")
        return None

def convert_document_to_pdf(input_file, output_dir=None):
    """
    Convert a document to PDF using available methods.
    Tries LibreOffice UNO API first, then falls back to unoconv.
    
    Args:
        input_file: Path to the input document
        output_dir: Directory to save the PDF (defaults to same directory as input)
        
    Returns:
        Path to the generated PDF or None if conversion failed
    """
    # Check if the file is already a PDF
    if input_file.lower().endswith('.pdf'):
        logger.info(f"File is already a PDF: {input_file}")
        return input_file
        
    # Try conversion with LibreOffice UNO API if available
    if UNO_AVAILABLE:
        pdf_path = convert_to_pdf_with_libreoffice(input_file, output_dir)
        if pdf_path:
            return pdf_path
    else:
        logger.info("python-uno not available, skipping LibreOffice UNO API conversion")
    
    # Try unoconv as fallback
    logger.info("Trying unoconv for PDF conversion")
    pdf_path = convert_to_pdf_with_unoconv(input_file, output_dir)
    
    return pdf_path


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
    Read the following Markdown and improve ONLY the formatting while preserving ALL original content exactly as written.

    1. Fix ONLY markdown formatting issues - do not change any words or fix any typos
    2. Make sure that headings descend in size logically (# > ## > ###)
    3. If you see raw URLs, change them into links with the EXACT SAME TEXT as link text
    4. Do Not Remove any Links (reformatting is ok)
    5. Format things that look like lists (ordered or unordered) into proper markdown lists
    6. Keep all original content including typos, repeated words, grammatical errors, etc.
    7. Do NOT make ANY content edits, not even minor ones like fixing duplicate words
    8. Do not add '```markdown' or '```' around your response

    !!! CRITICAL: DO NOT CHANGE ANY WORDS OR FIX ANY TYPOS, NO MATTER HOW OBVIOUS !!!
    !!! PRESERVE EXACT ORIGINAL CONTENT, INCLUDING ALL ERRORS AND DUPLICATED WORDS !!!

    Here is the content to improve:

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

def main():
    parser = argparse.ArgumentParser(description='Convert Word to Markdown and improve formatting')
    parser.add_argument('input_file', help='Path to input Word document')
    parser.add_argument('--output', help='Path to output Markdown file (default: input filename with .md extension)')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--mistral-api-key', help='Mistral API key for OCR')
    parser.add_argument('--skip-gpt', action='store_true', help='Skip GPT improvement step')
    parser.add_argument('--skip-html', action='store_true', help='Skip HTML conversion step')
    parser.add_argument('--show-diff', action='store_true', help='Show diff between original and improved markdown')
    parser.add_argument('--save-report', action='store_true', help='Save detailed comparison report to file')
    # By default, detailed comparison is enabled. To disable it, use --no-detailed-comparison.
    parser.add_argument('--no-detailed-comparison', dest='detailed_comparison', action='store_false', help='Disable detailed token-by-token comparison')
    parser.set_defaults(detailed_comparison=True)
    # New flag: by default, only the improved HTML file will be saved.
    # Use --keep-markdown to keep the intermediate markdown files.
    parser.add_argument('--keep-markdown', action='store_true', help='Keep the intermediate markdown files (default: do not keep them)')
    parser.add_argument('--use-pandoc', action='store_true', help='Use pandoc instead of OCR for document conversion')
    parser.add_argument('--use-libreoffice', action='store_true', help='Use LibreOffice to convert documents to PDF before OCR')
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
    
    # Check if the input is a document that needs conversion to PDF
    document_extensions = ['.doc', '.docx', '.odt', '.rtf', '.txt', '.wpd', '.wps']
    input_file = args.input_file
    
    if args.use_libreoffice and any(input_file.lower().endswith(ext) for ext in document_extensions):
        logger.info(f"Converting document to PDF using LibreOffice: {input_file}")
        pdf_path = convert_document_to_pdf(input_file)
        if pdf_path:
            # Now use the PDF with Mistral
            input_file = pdf_path
            logger.info(f"Using converted PDF for further processing: {input_file}")
        else:
            logger.error("Document conversion failed. Proceeding with original file.")
    
    # Determine conversion method
    use_ocr = not args.use_pandoc and mistral_api_key
    
    # Convert document to markdown
    if use_ocr:
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
        if args.use_pandoc:
            print(f"Converting {input_file} using pandoc (as requested)...")
        else:
            print(f"Converting {input_file} using pandoc (Mistral API key not provided)...")
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
    
    # If skip_gpt is True, use the original markdown
    if args.skip_gpt:
        print("Skipped GPT improvement.")
        final_markdown = original_markdown
    else:
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

    # Convert the final markdown (original or improved) to HTML using pypandoc.convert_text.
    # This will be the only output file if --keep-markdown is not used.
    if not args.skip_html:
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
    else:
        html_output = None

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

    if args.detailed_comparison and comparison_result['mismatches']:
        print("\n===== CONTENT DIFFERENCES =====")
        print(comparison_result['mismatch_report'])
    
    # No pandoc comparison - removed
    
    if args.show_diff:
        print("\n===== UNIFIED LINE DIFF =====")
        diff = difflib.unified_diff(
            original_markdown.splitlines(),
            final_markdown.splitlines(),
            lineterm='',
            n=3
        )
        print('\n'.join(diff))
    
    if comparison_result['mismatches']:
        print(f"\nFound {len(comparison_result['mismatches'])} content differences.")
        if not args.detailed_comparison:
            print("To see details, run without --no-detailed-comparison")
        print("\nIMPORTANT: Content differences should not occur! This needs investigation.")
    else:
        print("\nNo content differences detected. All words are preserved exactly in the final version.")

    if args.save_report:
        comparison_report_path = f"{os.path.splitext(args.input_file)[0]}_comparison_report.txt"
        try:
            with open(comparison_report_path, 'w', encoding='utf-8') as f:
                f.write("===== CONTENT COMPARISON REPORT =====\n\n")
                f.write("This report shows any content differences between the original and final versions\n\n")
                f.write(comparison_result['mismatch_report'])
                if comparison_result['missing_words']:
                    f.write("\n\n===== MISSING WORDS =====\n\n")
                    for word, count in comparison_result['missing_words'].items():
                        f.write(f"  - '{word}' appears {count} fewer times\n")
                if comparison_result.get('added_words'):
                    f.write("\n\n===== ADDED WORDS =====\n\n")
                    for word, count in comparison_result['added_words'].items():
                        f.write(f"  - '{word}' appears {count} more times\n")
                    
            print(f"\nDetailed comparison report saved to {comparison_report_path}")
        except Exception as e:
            print(f"Error writing comparison report: {e}")

if __name__ == "__main__":
    main()