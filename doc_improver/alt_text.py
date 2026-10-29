"""Alt text generation with the model for images that have none."""
import base64
import json
import re
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from colorama import Fore, Style
from tqdm import tqdm

from .images import (
    image_alt_text,
    image_media_type,
    image_target_path,
    resolve_image_file,
    restore_images,
)
from .llm import image_part, llm_request, text_part
from .md_syntax import IMAGE_ALT_RE, IMAGE_RE, find_headings
from .pricing import cost_summary, estimate_cost, print_cost_breakdown, sum_tokens
from .prompts import fill, load_prompt
from .report import format_tokens, plural

# Prompts for generating alt text, from config/prompts/alt_text.yaml. The image block is sent before the user prompt.
# In ALT_TEXT_USER_PROMPT, {context} is replaced with the text around the image (plain string replacement,
# so other braces are fine).
ALT_TEXT_SYSTEM_PROMPT = load_prompt("alt_text", "system")
ALT_TEXT_USER_PROMPT = load_prompt("alt_text", "user", slots=("context",))

# Structured output: the reply must be JSON with exactly one string field, so the model can't add
# explanations or reasoning around the alt text
ALT_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"alt_text": {"type": "string"}},
    "required": ["alt_text"],
    "additionalProperties": False,
}


def parse_alt_text_reply(reply):
    """Return (alt_text, problem) from the JSON reply of an alt text request; problem is None when it parsed."""
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as e:
        return None, f"the reply was not valid JSON ({e}); raw reply: {reply!r}"
    if not isinstance(data, dict) or not isinstance(data.get("alt_text"), str):
        return None, f"the reply had no alt_text string; raw reply: {reply!r}"
    return data["alt_text"], None


def build_alt_text_content(image_bytes, media_type, context):
    """Build the user message content for an alt text request (see llm.llm_request): the image, then the text."""
    return [
        image_part(media_type, base64.standard_b64encode(image_bytes).decode("utf-8")),
        text_part(fill(ALT_TEXT_USER_PROMPT, context=context)),
    ]


def image_context(markdown_text, image_start, image_end, min_words):
    """Return (context, heading) for an image. heading is the nearest heading line above the image, or None.

    The context is whole blocks (paragraphs, headings, list blocks; separated by blank lines) around the
    image: blocks are added going backward until at least min_words words precede the image, and going
    forward until at least min_words words follow it, crossing headings if needed. The nearest heading
    above is always included. Blocks that are only images are skipped. If the image sits inside a block
    with other text, that block is included with the image shown as [this image]. Other images are shown
    as [image]. Blocks are never cut, so the context can exceed min_words.
    """
    headings = find_headings(markdown_text)
    before_headings = [h for h in headings if h[0] <= image_start]
    heading = before_headings[-1][2] if before_headings else None

    def word_count(block):
        return len(IMAGE_RE.sub('', block).split())

    text_before = markdown_text[:image_start]
    text_after = markdown_text[image_end:]
    # The image's own block: the text on its side of the nearest blank lines
    own_before = re.split(r'\n[ \t]*\n', text_before)[-1]
    own_after = re.split(r'\n[ \t]*\n', text_after)[0]
    own_block = (own_before + "[this image]" + own_after).strip()
    blocks_before = [b.strip() for b in re.split(r'\n[ \t]*\n', text_before[:len(text_before) - len(own_before)])
                     if word_count(b)]
    blocks_after = [b.strip() for b in re.split(r'\n[ \t]*\n', text_after[len(own_after):]) if word_count(b)]

    own_words = word_count(own_block.replace("[this image]", ""))
    # Words in the image's own block count toward both sides
    needed = max(0, min_words - own_words)
    parts_before = []
    words = 0
    for block in reversed(blocks_before):
        if words >= needed:
            break
        parts_before.insert(0, block)
        words += word_count(block)
    parts_after = []
    words = 0
    for block in blocks_after:
        if words >= needed:
            break
        parts_after.append(block)
        words += word_count(block)

    parts = parts_before[:]
    if heading and heading not in parts:
        parts.insert(0, heading)
    if own_block != "[this image]":
        parts.append(own_block)
    parts.extend(parts_after)
    context = IMAGE_RE.sub("[image]", "\n\n".join(parts))
    return context, heading


def format_alt_text(alt_text):
    """Make generated alt text safe inside ![...]: whitespace runs (incl. newlines) become one space,
    and backslashes and square brackets are escaped."""
    alt_text = ' '.join(alt_text.split())
    return re.sub(r'([\\\[\]])', r'\\\1', alt_text)


def insert_alt_text(image_ref, alt_text):
    """Return the image reference with alt_text (already formatted) in place of its alt text."""
    alt_match = IMAGE_ALT_RE.match(image_ref)
    return f"![{alt_text}]" + image_ref[alt_match.end():]


@dataclass
class AltTextJob:
    """One image without alt text, in document order.

    lines collects everything printed about this image (its details, the call's own messages, its warning),
    so the lines can be printed in document order once all calls are done instead of interleaving.
    """
    number: int
    placeholder: str
    image_ref: str
    path: str = None
    heading: str = None
    warning: tuple = None
    future: object = None
    lines: list = field(default_factory=list)

    def warn(self, reason):
        self.warning = (self.image_ref, reason)
        self.lines.append(Fore.YELLOW + f"WARNING: no alt text generated for {self.image_ref}: {reason}. "
                          "Its alt text stays empty.")


class AltTextRun:
    """The alt text calls started by start_alt_text, running in a thread pool; finish_alt_text collects them."""

    def __init__(self, jobs, settings, prices, pool, verbose):
        self.jobs = jobs
        self.settings = settings
        self.prices = prices
        self.pool = pool
        self.verbose = verbose

    def cancel_pending(self):
        """Cancel the calls that have not started yet (running calls finish). Returns how many were cancelled."""
        return sum(1 for job in self.jobs if job.future is not None and job.future.cancel())

    def abort(self):
        """Cancel the calls not started yet and wait for the running ones, e.g. when the run is stopping on an
        exception. The results are not collected."""
        self.cancel_pending()
        self.pool.shutdown(wait=True, cancel_futures=True)


AltTextResult = namedtuple('AltTextResult', ['alt_by_placeholder', 'generated', 'warnings', 'costs'])


def start_alt_text(original_markdown, image_placeholders, media_base_dir, connection, settings, prices,
                   verbose=False):
    """Start an alt text call to the model at connection (see llm.llm_request) for every image whose alt text is
    empty; author-written alt text is kept.

    original_markdown is the converted text with the real image references, and image_placeholders the
    placeholder -> image reference mapping protect_images returned for it. Each image's context comes from
    original_markdown, so the calls don't wait for the formatting call. settings comes from
    config.get_alt_text_settings. The calls run in a thread pool of settings['concurrency'] threads; images
    that can't be sent (URL, missing or unreadable file) get a warning and no call. Nothing about the images is
    printed here except, with verbose, one header line; finish_alt_text prints the rest in document order.
    Returns an AltTextRun.
    """
    matches = list(IMAGE_RE.finditer(original_markdown))
    if [m.group(0) for m in matches] != list(image_placeholders.values()):
        raise ValueError("the image placeholders don't match the image references in the converted text, in order; "
                         "alt text can't be paired with its image")
    to_describe = [(placeholder, match) for placeholder, match in zip(image_placeholders, matches)
                   if not image_alt_text(match.group(0)).strip()]
    if verbose:
        print(f"\nAlt text: {len(to_describe)} of {len(matches)} image(s) have no alt text; generating it with "
              f"{settings['model']}, up to {settings['concurrency']} call(s) at a time, alongside the "
              "formatting call. Images that already have alt text are left unchanged.")

    pool = ThreadPoolExecutor(max_workers=settings['concurrency'], thread_name_prefix='alt-text')
    jobs = []
    try:
        for number, (placeholder, match) in enumerate(to_describe, 1):
            job = AltTextJob(number, placeholder, match.group(0))
            jobs.append(job)
            if verbose:
                job.lines.append(f"\nAlt text {number}/{len(to_describe)}: {job.image_ref}")
            path, problem = image_target_path(job.image_ref)
            if problem:
                job.warn(problem)
                continue
            job.path = path
            file_path = resolve_image_file(path, media_base_dir)
            if not file_path:
                job.warn(f"file not found (looked for {path} relative to {media_base_dir})")
                continue
            try:
                media_type = image_media_type(file_path)
            except ValueError as e:
                job.warn(str(e))
                continue
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            context, job.heading = image_context(original_markdown, match.start(), match.end(),
                                                 settings['context_words'])
            if verbose:
                job.lines.append(f"Sending {path} ({media_type}, {len(image_bytes)} bytes) with this surrounding "
                                 f"text (nearest heading, at least {settings['context_words']} words before "
                                 "and after, in whole blocks):")
                job.lines.append(Style.DIM + (context if context
                                              else "(none: no heading or text paragraphs near this image)"))
            content = build_alt_text_content(image_bytes, media_type, context)
            job.future = pool.submit(llm_request, connection, ALT_TEXT_SYSTEM_PROMPT, content,
                                     model=settings['model'], max_tokens=settings['max_tokens'],
                                     effort=settings['effort'], label=f"Alt text {number}/{len(to_describe)}",
                                     verbose=verbose, json_schema=ALT_TEXT_SCHEMA, show_progress=False,
                                     log=job.lines.append)
    except BaseException:
        AltTextRun(jobs, settings, prices, pool, verbose).abort()
        raise
    return AltTextRun(jobs, settings, prices, pool, verbose)


def finish_alt_text(run, stop_reason=None):
    """Wait for the alt text calls, then print each image's lines in document order and one summary line.

    stop_reason, if given, says why calls were cancelled (see AltTextRun.cancel_pending); cancelled calls are
    counted on the summary line. Every finished call is priced, including calls whose alt text won't be used.
    Returns an AltTextResult: alt_by_placeholder maps each image's placeholder to its generated (formatted) alt
    text, generated is a list of (image path, alt text, nearest heading) and warnings a list of (image
    reference, reason), both in document order, and costs a list of pricing.CallCost, one per finished call.
    """
    jobs = run.jobs
    settings = run.settings
    try:
        pending = [job.future for job in jobs if job.future is not None and not job.future.done()]
        if pending:
            # Only after the formatting call is done, so this bar never shares the screen with its counter
            with tqdm(total=len(pending), desc="Alt text: waiting for the calls", unit=" call",
                      leave=False) as progress:
                for _ in as_completed(pending):
                    progress.update(1)
    finally:
        run.pool.shutdown(wait=True, cancel_futures=True)

    alt_by_placeholder = {}
    generated = []
    warnings = []
    # Token usage and estimated cost of every finished call; calls without usage are counted separately
    call_tokens = []
    costs = []
    calls_without_usage = 0
    cancelled = 0
    for job in jobs:
        if job.future is not None and job.future.cancelled():
            cancelled += 1
            if run.verbose:
                job.lines.append(f"Not sent: cancelled ({stop_reason}).")
        elif job.future is not None:
            reply, tokens = job.future.result()
            costs.append(estimate_cost(settings['model'], tokens, run.prices))
            if tokens is None:
                calls_without_usage += 1
            else:
                call_tokens.append(tokens)
            alt_text, problem = None, None
            if not reply or not reply.strip():
                problem = "the API call failed or returned no text (see the error above)"
            else:
                alt_text, problem = parse_alt_text_reply(reply)
                if not problem and not alt_text.strip():
                    problem = "the model returned an empty alt_text"
            if problem:
                job.warn(problem)
            else:
                formatted = format_alt_text(alt_text)
                alt_by_placeholder[job.placeholder] = formatted
                generated.append((job.path, formatted, job.heading))
                if run.verbose:
                    job.lines.append(Fore.CYAN + f"Generated alt text for {job.path}: {formatted}")
        if job.warning:
            warnings.append(job.warning)
        for line in job.lines:
            print(line)

    # One summary line; the alt text itself is listed in the ADDED TEXT report at the end of the run
    effort = settings['effort'] or 'API default'
    if cancelled:
        result = (f"stopped ({stop_reason}): {plural(cancelled, 'call')} cancelled before sending, "
                  f"{len(generated)} of {len(jobs)} got alt text")
    elif len(generated) == len(jobs):
        result = "done"
    else:
        result = f"done, {len(generated)} of {len(jobs)} got alt text (see the warnings above)"
    usage = format_tokens(sum_tokens(call_tokens))
    if calls_without_usage:
        usage += f" ({plural(calls_without_usage, 'call')} returned no usage)"
    blank_line = "\n" if run.verbose else ""
    print(f"{blank_line}Alt text with {settings['model']} (effort {effort}): "
          f"{plural(len(jobs), 'image')}... {result}: {usage}{cost_summary(costs)}")
    if run.verbose:
        print_cost_breakdown(costs)
    return AltTextResult(alt_by_placeholder, generated, warnings, costs)


def restore_images_with_alt_text(text, image_placeholders, alt_by_placeholder):
    """Put the image references back in place of their placeholders, each with its generated alt text if it has
    one. The alt text is paired by placeholder, so it lands on the right image even if the model moved images."""
    with_alt = {placeholder: (insert_alt_text(image_ref, alt_by_placeholder[placeholder])
                              if placeholder in alt_by_placeholder else image_ref)
                for placeholder, image_ref in image_placeholders.items()}
    return restore_images(text, with_alt)
