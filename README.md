# Anything to Html Converter

A tool for converting documents (PDF, Word, etc.) to web-friendly HTML with content integrity preservation.

## ⚠️ EARLY BETA ⚠️

**This tool is in very early beta stages. Expect bugs, incomplete features, and potential issues.**

## Overview

Anything to Html Converter solves the common workplace problem of converting document files (like Word documents or PDFs) into clean HTML for website publishing. It's particularly useful for when you receive formatted documents from cowokers that need to be published online.

The tool:
1. Converts documents to markdown using either Pandoc or OCR (via Mistral AI)
2. Improves markdown formatting using GPT-4o-mini
3. Converts the markdown to clean HTML
4. Verifies that no content was lost during conversion
5. Generates detailed comparison reports

## Requirements

- Python 3.6+
- Pandoc (for document conversion)
- Optional: Mistral AI API key (for OCR-based conversion)
- Optional: OpenAI API key (for GPT formatting improvements)
- Optional: LibreOffice (for improved document conversion)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anything-to-html.git
cd anything-to-html

# Install dependencies
pip install -r requirements.txt

# Install Pandoc (if not already installed)
# On Ubuntu/Debian:
apt-get install pandoc

# On macOS:
brew install pandoc

# On Windows:
# Download from https://pandoc.org/installing.html
```

## Usage

Basic usage:

```bash
python converter.py input_document.docx
```


## Command Line Options

```
positional arguments:
  input_file             Path to input document

optional arguments:
  --output OUTPUT           Path to output Markdown file (default: input filename with .html extension)
  --api-key API_KEY         OpenAI API key
  --mistral-api-key KEY     Mistral API key for OCR
  --skip-gpt                Skip GPT improvement step
  --skip-html               Skip HTML conversion step
  --show-diff               Show diff between original and improved markdown
  --save-report             Save detailed comparison report to file
  --no-detailed-comparison  Disable detailed token-by-token comparison
  --keep-markdown           Keep the intermediate markdown files (default: do not keep them)
  --use-pandoc              Use pandoc instead of OCR for document conversion
  --use-libreoffice         Use LibreOffice to convert documents to PDF before OCR
  --debug                   Enable debug logging
```

## Conversion Methods

The tool supports three primary conversion methods:

1. **Pandoc conversion** (default): Uses Pandoc to convert documents to markdown
2. **OCR conversion** (with `--mistral-api-key`): Uses Mistral AI's OCR service for more accurate extraction from complex documents
3. **LibreOffice pre-processing** (with `--use-libreoffice`): Converts documents to PDF before OCR processing

## API Keys and Environment Variables

**NOTE: By default, the tool expects API keys to be set as environment variables rather than passed as command-line arguments.**

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY=your_openai_api_key
export MISTRAL_API_KEY=your_mistral_api_key
```


## Content Preservation Verification

The tool includes a detailed verification system to ensure no content is lost during conversion:

- Word-level comparison between original and converted document
- Detailed reporting of any content differences
- Verification of content integrity across conversion steps