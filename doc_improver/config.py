"""Settings: loading .env, the connection to the model, and the formatting and alt text settings."""
import os

from dotenv import load_dotenv

# Load settings from the .env file in the project folder (the folder holding anything-to-anything.py).
# A setting already in the environment wins over .env. If .env sets a name twice, the last one is used.
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(ENV_PATH)

# The API styles an endpoint can speak (llm.API_STYLES holds the request function for each)
API_STYLE_NAMES = ('anthropic', 'openai')


def whole_number_setting(value, env_name):
    """Return (number, error) for a whole-number setting read from env_name in .env."""
    try:
        return int(value), None
    except ValueError:
        return None, f"{env_name} in .env must be a whole number, got {value!r}"


def get_connection(args, parser):
    """Return the connection settings {api, base_url, api_key}; errors exit through parser.error.

    api is --api, else LLM_API. base_url is --base-url, else LLM_BASE_URL; blank means the API's official
    endpoint. The key is LLM_API_KEY, which is required (a local server that needs no key accepts any text).
    """
    if not args.api:
        parser.error(f"no API style given: set LLM_API in .env or pass --api ({' or '.join(API_STYLE_NAMES)})")
    if args.api not in API_STYLE_NAMES:
        parser.error(f"LLM_API in .env is {args.api!r}; expected one of: {', '.join(API_STYLE_NAMES)}")
    api_key = os.getenv('LLM_API_KEY')
    if not api_key:
        parser.error("no API key: set LLM_API_KEY in .env (for a local server that needs no key, any text works)")
    return {'api': args.api, 'base_url': args.base_url or None, 'api_key': api_key}


def resolve_format_settings(args, parser):
    """Fill in args.model, args.max_tokens and args.effort from FORMAT_MODEL, FORMAT_MAX_TOKENS and
    FORMAT_EFFORT where the command line left them out; errors exit through parser.error."""
    if not args.model:
        args.model = os.getenv('FORMAT_MODEL')
    if not args.model:
        parser.error("no model given: set FORMAT_MODEL in .env or pass --model")
    if args.max_tokens is None:
        env_max_tokens = os.getenv('FORMAT_MAX_TOKENS')
        if not env_max_tokens:
            parser.error("no max tokens given: set FORMAT_MAX_TOKENS in .env or pass --max-tokens")
        args.max_tokens, error = whole_number_setting(env_max_tokens, 'FORMAT_MAX_TOKENS')
        if error:
            parser.error(error)
    if args.max_tokens <= 0:
        parser.error(f"max tokens must be a positive number, got {args.max_tokens}")
    if not args.effort:
        args.effort = os.getenv('FORMAT_EFFORT') or None


def get_alt_text_settings(args):
    """Return the alt text settings, {model, max_tokens, effort, context_words, concurrency}, or an error
    message string.

    The model is --alt-text-model, else ALT_TEXT_MODEL, else the formatting model (args.model).
    Max tokens and effort are separate from the formatting call: --alt-text-max-tokens / --alt-text-effort,
    defaulting to ALT_TEXT_MAX_TOKENS / ALT_TEXT_EFFORT. context_words is the minimum number of words of
    surrounding text sent before and after each image (--alt-text-context-words, else ALT_TEXT_CONTEXT_WORDS).
    concurrency is the most alt text calls running at once (the formatting call is not counted):
    --alt-text-concurrency, defaulting to ALT_TEXT_CONCURRENCY.
    """
    advice = "or pass --no-alt-text"
    model = args.alt_text_model or os.getenv('ALT_TEXT_MODEL') or args.model
    max_tokens = args.alt_text_max_tokens
    if max_tokens is None:
        env_max_tokens = os.getenv('ALT_TEXT_MAX_TOKENS')
        if not env_max_tokens:
            return ("alt text needs a max tokens setting: set ALT_TEXT_MAX_TOKENS in .env, "
                    f"pass --alt-text-max-tokens {advice}")
        max_tokens, error = whole_number_setting(env_max_tokens, 'ALT_TEXT_MAX_TOKENS')
        if error:
            return error
    if max_tokens <= 0:
        return f"alt text max tokens must be a positive number, got {max_tokens}"
    effort = args.alt_text_effort or os.getenv('ALT_TEXT_EFFORT') or None
    context_words = args.alt_text_context_words
    if context_words is None:
        env_context_words = os.getenv('ALT_TEXT_CONTEXT_WORDS')
        if not env_context_words:
            return ("alt text needs a context size: set ALT_TEXT_CONTEXT_WORDS in .env, "
                    f"pass --alt-text-context-words {advice}")
        context_words, error = whole_number_setting(env_context_words, 'ALT_TEXT_CONTEXT_WORDS')
        if error:
            return error
    if context_words < 0:
        return f"alt text context words can't be negative, got {context_words}"
    concurrency = args.alt_text_concurrency
    if concurrency is None:
        env_concurrency = os.getenv('ALT_TEXT_CONCURRENCY')
        if not env_concurrency:
            return ("alt text needs a concurrency limit: set ALT_TEXT_CONCURRENCY in .env, "
                    f"pass --alt-text-concurrency {advice}")
        concurrency, error = whole_number_setting(env_concurrency, 'ALT_TEXT_CONCURRENCY')
        if error:
            return error
    if concurrency < 1:
        return f"alt text concurrency must be at least 1, got {concurrency}"
    return {'model': model, 'max_tokens': max_tokens, 'effort': effort, 'context_words': context_words,
            'concurrency': concurrency}
