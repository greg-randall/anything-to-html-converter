# Anything to HTML Converter

**Project URL:** [https://github.com/greg-randall/anything-to-html-converter](https://github.com/greg-randall/anything-to-html-converter)

A tool for converting various document formats (including Word documents and OCR-capable image-based files) into clean, web-friendly HTML, with a strong emphasis on preserving content integrity.

## Overview

The Anything to HTML Converter simplifies transforming documents into HTML suitable for website publishing. It's designed to maintain the original content as accurately as possible throughout the conversion process.

The tool follows these key steps:

1.  **Initial Conversion to Markdown:** Input documents (e.g., `.docx`, PDFs, images) are first converted to Markdown.
    * If a Mistral API key is provided, the script prioritizes Mistral AI for its OCR capabilities, which can handle a wider range of file types.
    * If a Mistral API key is not available, or if OCR processing fails, the script falls back to using Pandoc for conversion (primarily for formats like `.docx`).
2.  **Markdown Refinement (Optional but Recommended):** The generated Markdown can be processed by OpenAI's GPT-4o-mini model (requires an OpenAI API key). This step focuses *exclusively* on improving formatting – normalizing headings, lists, tables, and converting raw URLs to clickable links – **without altering the textual content, including typos or original phrasing.**
3.  **Conversion to HTML:** The final Markdown (either the original or the GPT-improved version) is converted into a standalone HTML file using Pandoc.
4.  **Content Integrity Check:** A rigorous comparison is performed between the Markdown content *before* GPT refinement and *after* GPT refinement (if applied). This step tokenizes the text and identifies any changes, additions, or deletions to the actual content, helping to ensure that the LLM has only adjusted formatting.
5.  **Reporting:** Detailed reports are generated if any content discrepancies are found during the integrity check.

### Why Markdown as an Intermediary?

Large Language Models (LLMs) like GPT are incredibly powerful for text processing. However, when asked to directly edit or reformat HTML, they can sometimes misinterpret the structure and inadvertently alter the content. HTML's nested tags and attributes present a complex environment where content and presentation are tightly coupled.

Markdown, with its simpler, content-focused syntax, serves as a more robust intermediary. By converting the document to Markdown first, we can leverage an LLM to:
* Focus on common formatting issues (like inconsistent headings, messy lists, or plain URLs that should be links).
* Perform these tasks with a higher degree of accuracy in preserving the *actual text*.

The script then uses Pandoc, a deterministic conversion tool, for the final, reliable transformation from refined Markdown to HTML. This multi-step process, with Markdown at its core, allows for sophisticated formatting improvements while minimizing the risk of unintended content changes.

## Requirements

* **Python 3.x**
* **Pandoc:** Must be installed and accessible in your system's PATH. Pandoc is used for `.docx` to Markdown conversion (as a fallback or primary method if not using OCR via Mistral) and for the final Markdown to HTML conversion.
* **Python Libraries:**
    * `openai` (for GPT-based Markdown improvement)
    * `pypandoc` (Python wrapper for Pandoc)
    * `mistralai` (for OCR-based document conversion using Mistral AI)
    * `httpx` (dependency for AI SDKs)
    * `markdown` (used for an internal utility function for basic Markdown to HTML conversion, though the main pipeline uses `pypandoc` for richer, standalone HTML output)
    * Standard libraries used: `os`, `re`, `argparse`, `collections.Counter`, `difflib`, `time`, `logging`. These are typically included with Python.

## Installation

1.  **Clone the repository or download the script:**
    ```bash
    git clone [https://github.com/greg-randall/anything-to-html-converter.git](https://github.com/greg-randall/anything-to-html-converter.git)
    cd anything-to-html-converter
    ```
    Or, if you just have the `anything-to-html-converter.py` file, place it in your desired directory.

2.  **Install Pandoc:**
    * **On Ubuntu/Debian:**
        ```bash
        sudo apt-get update
        sudo apt-get install pandoc
        ```
    * **On macOS (using Homebrew):**
        ```bash
        brew install pandoc
        ```
    * **On Windows:**
        Download the installer from [pandoc.org/installing.html](https://pandoc.org/installing.html).

3.  **Install required Python libraries:**
    Create a `requirements.txt` file in the same directory as the script with the following content:
    ```txt
    openai
    pypandoc
    mistralai
    httpx
    markdown
    ```
    Then, install these libraries using pip (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Keys (Optional but Recommended):**
    * **OpenAI API Key:** For Markdown improvement using GPT-4o-mini.
    * **Mistral API Key:** For using Mistral AI's OCR capabilities.
    You can provide these keys via command-line arguments (`--api-key` for OpenAI, `--mistral-api-key` for Mistral) or by setting them as environment variables (`OPENAI_API_KEY`, `MISTRAL_API_KEY`).

## Usage

### Basic Command
To convert a document, run the script with the path to your input file:
```bash
python anything-to-html-converter.py your_document.docx