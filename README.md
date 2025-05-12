# Anything to HTML Converter

A tool for converting documents (primarily Word documents, with OCR capabilities for other formats) to web-friendly HTML, focusing on content integrity.

## Overview

The Anything to HTML Converter aims to simplify the process of converting document files (like Word documents) into clean HTML suitable for website publishing. It's particularly useful when you need to publish documents online while preserving the original content as accurately as possible.

The tool performs the following key steps:
1.  Converts input documents (e.g., `.docx`) to Markdown. It prioritizes using Mistral AI for OCR if a Mistral API key is provided. If not, or if OCR fails, it falls back to using Pandoc for conversion.
2.  Optionally improves the Markdown formatting using OpenAI's GPT-4o-mini model (if an OpenAI API key is provided and the step is not skipped). The improvement focuses on formatting normalization (headings, lists, tables, URL-to-link conversion) **without altering the textual content**.
3.  Converts the final Markdown (either original or GPT-improved) to a standalone HTML file using Pandoc.
4.  Performs a detailed comparison between the original Markdown (pre-GPT improvement) and the final Markdown (post-GPT improvement, if applied) to verify that no textual content was inadvertently changed, added, or deleted.
5.  Generates reports on content differences, if any are found.

## Requirements

-   Python 3.x
-   Pandoc (must be installed and in the system's PATH)
-   Python libraries:
    -   `os`
    -   `re`
    -   `argparse`
    -   `collections` (specifically `Counter`)
    -   `openai`
    -   `difflib`
    -   `pypandoc`
    -   `time`
    -   `mistralai`
    -   `httpx`
    -   `logging`
    -   `markdown` (for an internal step, though final HTML conversion uses `pypandoc`)

## Installation

1.  **Clone the repository (if applicable) or download the script.**
    ```bash
    # If it's a git repository:
    # git clone [https://github.com/yourusername/anything-to-html.git](https://github.com/yourusername/anything-to-html.git)
    # cd anything-to-html
    ```

2.  **Install Pandoc:**
    -   **On Ubuntu/Debian:**
        ```bash
        sudo apt-get install pandoc
        ```
    -   **On macOS (using Homebrew):**
        ```bash
        brew install pandoc
        ```
    -   **On Windows:**
        Download the installer from [pandoc.org](https://pandoc.org/installing.html).

3.  **Install required Python libraries:**
    Create a `requirements.txt` file with the following content:
    ```txt
    openai
    pypandoc
    mistralai
    httpx
    markdown
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Basic usage:
```bash
python anything-to-html-converter.py your_document.docx