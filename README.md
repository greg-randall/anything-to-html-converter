# anything-to-anything

Turns a document (Word, HTML, RTF, a spreadsheet and many more) into clean, accessible Markdown and HTML.

A language model rewrites the formatting. It fixes headings, lists and link text, and writes alt text for images that have none. Then the script checks that no content was lost along the way.

```
$ python3 anything-to-anything.py example.docx
Converting example.docx to markdown... done (3 images protected)
Formatting with claude-sonnet-5-5 (effort high)... done: 4,200 in / 4,350 out tokens, $0.0519
Alt text with claude-sonnet-5-5 (effort low): 3 images... done: 3,000 in / 100 out tokens, $0.0070
Total estimated cost (list price): $0.0589
Saved example_improved.md and example_improved.html

All link URLs preserved. All original words preserved.
Link text edits (2, expected): "here" -> "" (x2)

===== ADDED TEXT - PLEASE CHECK: 3 alt text description(s) were written by the model and added to the document =====

  Image:   example_improved_media/image1.jpeg
  Alt:     Volunteers planting trees along a riverbank during the spring cleanup

  Image:   example_improved_media/image2.png
  Alt:     Bar chart of library visits per month, rising from January to June

  Image:   example_improved_media/image3.png
  Alt:     City Library logo

Review each description for accuracy before publishing.
```

In the saved Markdown, each of those images now carries its alt text, for example:

```markdown
![Volunteers planting trees along a riverbank during the spring cleanup](example_improved_media/image1.jpeg){width="3in" height="2in"}
```

## What it does

1. **Converts** the input to Markdown with [pandoc](https://pandoc.org).
   - Spreadsheets (`.xlsx`, `.xls`) are read with pandas instead. Only single-sheet workbooks are accepted, so no sheet is silently dropped.
   - Images are extracted into a folder named after the output, e.g. `example_improved_media/`, so two documents never overwrite each other's images.
2. **Protects images.** Every image reference is replaced with a placeholder before the text goes to the model, and put back exactly as it was afterwards, with its size and path intact. If the model drops, duplicates or changes a placeholder, the run stops and writes nothing.
3. **Formats** the text with the model.
   - It normalizes headings and turns lists into real lists.
   - It rewrites links so the link text describes the destination. "click here" becomes a meaningful link for screen readers.
   - The prompt forbids changing, adding or removing words, apart from those link edits.
4. **Writes alt text** for every image that has none, while the formatting call runs. Alt text an author already wrote is never touched. See [Alt text](#alt-text).
5. **Saves** `<name>_improved.md` and `<name>_improved.html` (a standalone page with a UTF-8 charset).
6. **Checks the result against the original:**
   - every URL is still present and unchanged
   - every word is still there (markup such as `2^nd^` vs `2nd` is ignored)
   - word-level differences, each shown with its surrounding context
   - expected link-text edits are listed separately, so they don't look like errors
   - list numbers the model added (plain lines turned into a numbered list) or removed are listed separately too, so they don't count as added or missing words. A numbered list whose numbers don't count up by one (1, 2, 5) is a warning, and its numbers stay in the added words
7. **Lists everything the model added.** The last thing on screen is the "ADDED TEXT - PLEASE CHECK" section, which shows each generated alt text so a person can check it before publishing.

Problems are never hidden: errors, refusals, cut-off responses, changed URLs, missing words and failed images always print in full, even without `--verbose`.

## Requirements

- **Python 3**
- **pandoc** on your `PATH`, e.g. `sudo apt install pandoc`
- **Python packages:**
  ```
  python3 -m pip install anthropic openai pypandoc pandas openpyxl pillow colorama tqdm python-dotenv pyyaml
  ```
  For the tests, also install `pytest python-docx`.
- **A model endpoint** that speaks either:
  - the **Anthropic** Messages API: Claude, or a server with an Anthropic-style endpoint, or
  - the **OpenAI** Chat Completions API: OpenAI, or any OpenAI-compatible server (Ollama, LM Studio, vLLM, OpenRouter, Azure OpenAI and many more).

  For alt text, the model must accept images.

## Setup

Copy `blank.env` to `.env` in the project folder and put your API key in `LLM_API_KEY`. Everything else is filled in, and each setting is explained in the file.

- It is set up for Anthropic (`claude-sonnet-5-5`).
- To use OpenAI (`gpt-6.1-terra`), comment out the Anthropic block and uncomment the OpenAI block.
- For any other server, set `LLM_API`, `LLM_BASE_URL` and the model names yourself (see below).

`.env` is git-ignored, so keep keys there and nowhere else. `blank.env` is tracked, so never put a key in it.

`LLM_BASE_URL` examples:

| Endpoint | `LLM_API` | `LLM_BASE_URL` |
|---|---|---|
| Anthropic | `anthropic` | blank |
| OpenAI | `openai` | blank |
| Claude on Azure AI Foundry | `anthropic` | `https://<resource>.services.ai.azure.com/anthropic/` |
| Azure OpenAI | `openai` | `https://<resource>.openai.azure.com/openai/v1/` |
| Ollama | `openai` | `http://localhost:11434/v1` |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` |

The model name is whatever the endpoint calls it, e.g. the deployment name on Azure. A local server that needs no key still needs some text in `LLM_API_KEY`.

A setting that is already set in your shell's environment wins over `.env`.

Nothing is hardcoded. If a setting the run needs is missing, the script stops and names it.

## Usage

```
python3 anything-to-anything.py <input file> [options]
```

| Option | What it does |
|---|---|
| `--output PATH` | Output `.html` or `.md` path. The other format is saved next to it with the same name. Default: `<input>_improved.html` |
| `--api {anthropic,openai}` | API style of the endpoint. Default: `LLM_API` |
| `--base-url URL` | Endpoint base URL. Default: `LLM_BASE_URL` |
| `--model NAME` | Formatting model. Default: `FORMAT_MODEL` |
| `--max-tokens N` | Max output tokens for formatting |
| `--effort LEVEL` | Effort for formatting (Anthropic effort or OpenAI reasoning_effort) |
| `--no-alt-text` | Skip alt text generation |
| `--alt-text-model NAME` | Model for alt text |
| `--alt-text-max-tokens N` | Max output tokens per alt text call |
| `--alt-text-effort LEVEL` | Effort for alt text |
| `--alt-text-context-words N` | Minimum words of surrounding text sent with each image |
| `--alt-text-concurrency N` | Most alt text calls running at once |
| `--verbose` | Show the details of each call: endpoint, model, full token usage, the text sent with each image, and a cost breakdown |
| `--show-diff` | Also print a unified line diff of original vs improved |
| `--no-detailed-comparison` | Show only the count of content differences |
| `--keep-markdown` | Keep the intermediate pandoc Markdown file |

Every option that has a `.env` setting overrides it for that run only. Examples:

```bash
# One run with a different alt text model
python3 anything-to-anything.py example.docx --alt-text-model claude-haiku-5-5

# Use a local Ollama model and skip alt text
python3 anything-to-anything.py page.html --api openai --base-url http://localhost:11434/v1 \
    --model llama3.3 --no-alt-text

# Write the result somewhere else
python3 anything-to-anything.py report.rtf --output out/report.html
```

Inputs pandoc can't read (for example `.pdf`) are rejected with the converter's error message, and nothing is written.

## Alt text

For each image without alt text, the model receives:
- the image itself
- the nearest heading above it
- surrounding paragraphs: at least `ALT_TEXT_CONTEXT_WORDS` words before and after, as whole paragraphs, never cut

The heading and paragraphs come from the original converted text, not the formatted text. That way the alt text calls don't wait for the formatting call.

The calls run in parallel, with each other and with the formatting call:
- At most `ALT_TEXT_CONCURRENCY` alt text calls run at once. The formatting call is not counted.
- Each alt text is paired with its own image, so it stays correct even if the model moves images around.
- It is inserted only after the image checks pass.
- Warnings and `--verbose` details for each image print after all calls finish, in document order.
- If the formatting call fails, alt text calls that haven't started are cancelled. Finished calls still count in the cost total, and nothing is written.

The prompt is in `config/prompts/alt_text.yaml`. Its rules include:
- Be brief: one short sentence, under 100 characters.
- Cover only the main subject. **Never describe clothing or accessories.**
- Name a person only if the surrounding text clearly identifies who is pictured. Never guess names.
- **Never mention or imply race, ethnicity, age, health, sexual orientation, gender, or any other personal attribute.** Say "person" or "people".

The reply is requested as structured JSON (`{"alt_text": "..."}`), so the model cannot wrap explanations or reasoning around the answer. A reply that doesn't parse produces a warning and leaves that image's alt text empty.

Generated alt text appears as the image's `alt` attribute only. It is not shown as a visible caption.

**Always review the "ADDED TEXT - PLEASE CHECK" list before publishing.** Models misread text in images and can break rules.

## Cost

Each model call prints an estimated cost, and the run ends with a total. The APIs return token counts, not prices, so the cost is worked out from `config/prices.yaml`. It holds the USD price per 1M tokens for each model name, with a note on where each price came from. It covers both default models in `blank.env` (`claude-sonnet-5-5` and `gpt-6.1-terra`) and `claude-haiku-5-5` and `claude-opus-5-5`; add an entry for any other model you use.

- Thinking tokens are counted as output tokens, as the API bills them.
- A model with no price shows `cost unknown` instead of a guess. So does a server that reports no token usage.
- These are list prices. Your actual bill may include discounts or credits.
- `--verbose` adds a per-token-type breakdown.

## Configuration files

```
config/
  prices.yaml          # USD per 1M tokens, per model
  prompts/
    format.yaml        # the formatting prompt
    alt_text.yaml      # the alt text prompt
```

Edit the prompts here, not in the code. Slots like `{context}` are filled by plain text replacement, so other braces and backticks are sent exactly as written. A malformed file stops the run before any paid call, with an error naming the file and key.

## Project layout

```
anything-to-anything.py   # launcher
doc_improver/
  cli.py                  # command-line options and the run, step by step
  config.py               # .env loading, connection and model settings
  convert.py              # pandoc and spreadsheet conversion, Markdown to HTML
  images.py               # image placeholders and image file lookup
  alt_text.py             # alt text context, request and insertion
  formatting.py           # the formatting request
  llm.py                  # one request function for both API styles
  llm_anthropic.py        # Anthropic-style requests
  llm_openai.py           # OpenAI-style requests
  links.py                # link and URL extraction and comparison
  lists.py                # numbered list detection, so list numbers aren't counted as words
  compare.py              # word comparison between original and improved text
  md_syntax.py            # shared Markdown patterns and tokenizer
  pricing.py              # cost estimates from prices.yaml
  prompts.py              # YAML prompt loading
  report.py               # console reports
config/                   # prices and prompts (see above)
tests/                    # pytest suite
```

## Tests

```
python3 -m pytest -q
```

The suite runs offline in a couple of seconds:
- Both model calls are replaced with fakes, so tests never call an API and cost nothing.
- Prompt tests read `config/prompts/*.yaml` and check that the document text, image rule and context are filled in correctly, so you can edit the prompts without touching the tests.
- End-to-end tests run the whole pipeline on tiny documents in a temporary folder.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `--- Not Found ---` | `LLM_BASE_URL` or the model name is wrong. Many OpenAI-compatible servers need `/v1` at the end of the base URL |
| A call fails for one model but not another | The endpoint has no model under exactly that name |
| `--- Bad request ---` on alt text calls | The model doesn't accept images or JSON schema output. Pick another `ALT_TEXT_MODEL` or pass `--no-alt-text` |
| `--- Bad request ---` mentioning `max_completion_tokens` or `reasoning_effort` | The OpenAI-compatible server doesn't support that parameter. Leave the effort blank, or try another server |
| `cost unknown (no price for X in prices.yaml)` | Add an entry for model `X` to `config/prices.yaml` |
| "pandoc could not find N image(s)" | An HTML/Markdown input links to images that don't exist relative to the input file |
| Need to see what was sent to the model | Re-run with `--verbose` |

## License

AGPL-3.0. See [LICENSE](LICENSE). You may use, change and share this project; if you distribute a modified version, or run one for others to use over a network, you must release its source under the same license.
