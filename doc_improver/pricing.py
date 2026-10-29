"""Estimated cost of the LLM calls: token counts times the list prices in prices.yaml.

The APIs return token counts only, so the cost is an estimate at list price. config/prices.yaml (in the project
folder, next to config/prompts/) maps each model name to USD per 1M tokens for input, output, cache write and
cache read, plus an optional threshold: when a request's input tokens (uncached input + cache write + cache
read) exceed threshold_input_tokens, the above_threshold prices apply to that whole request.

A model missing from prices.yaml, or a blank price for a token type the request used, is not an error:
the cost of that request is shown as unknown. A malformed file is an error that names the file and key.
"""
import functools
import numbers
import os
from dataclasses import dataclass

import yaml

# config/prices.yaml, in the project folder next to this package (with the prompts in config/prompts)
PRICES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "prices.yaml")
PRICES_NAME = "prices.yaml"

# The token types a call reports, in the order they are shown. Thinking tokens are part of output.
TOKEN_TYPES = ('input', 'output', 'cache_write', 'cache_read')
TOKEN_LABELS = {'input': 'input', 'output': 'output', 'cache_write': 'cache write', 'cache_read': 'cache read'}
ENTRY_KEYS = {'source', 'threshold_input_tokens', 'above_threshold', *TOKEN_TYPES}


def make_tokens(input=0, output=0, cache_write=0, cache_read=0):
    """Return a call's token counts as a dict with every token type. input is the uncached input."""
    return {'input': input, 'output': output, 'cache_write': cache_write, 'cache_read': cache_read}


def sum_tokens(token_dicts):
    """Add up token dicts (None entries, calls without usage, must be left out by the caller)."""
    return make_tokens(**{t: sum(tokens[t] for tokens in token_dicts) for t in TOKEN_TYPES})


def _check_prices(path, where, prices, allowed):
    """Check a dict of prices: only allowed keys; each price blank (None) or a number >= 0."""
    if not isinstance(prices, dict):
        raise TypeError(f"{where} in {path} must be a mapping of price names to prices, "
                        f"got {type(prices).__name__}")
    for key, value in prices.items():
        if key not in allowed:
            raise KeyError(f"{where} in {path} has an unknown key {key!r}; expected only: "
                           f"{', '.join(sorted(allowed))}")
        is_price = isinstance(value, numbers.Real) and not isinstance(value, bool) and value >= 0
        if key in TOKEN_TYPES and value is not None and not is_price:
            raise TypeError(f"{where}, key {key!r} in {path} must be a price in USD per 1M tokens (a number, "
                            f"0 or more) or blank, got {value!r}")


def _check_entry(path, name, entry):
    where = f"Entry {name!r}"
    _check_prices(path, where, entry, ENTRY_KEYS)
    source = entry.get('source')
    if source is not None and not isinstance(source, str):
        raise TypeError(f"{where}, key 'source' in {path} must be text, got {source!r}")
    threshold = entry.get('threshold_input_tokens')
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0):
        raise TypeError(f"{where}, key 'threshold_input_tokens' in {path} must be a whole number of tokens "
                        f"above 0, got {threshold!r}")
    if 'above_threshold' in entry:
        if threshold is None:
            raise KeyError(f"{where} in {path} has 'above_threshold' prices but no 'threshold_input_tokens'")
        above = entry['above_threshold']
        if above is not None:
            _check_prices(path, f"{where}, key 'above_threshold'", above, set(TOKEN_TYPES))


@functools.cache
def _load_file(path):
    """Return the checked prices of one YAML file as a dict (read once, then cached)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Price file {path} is missing")
    with open(path, encoding="utf-8") as f:
        try:
            prices = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Price file {path} is not valid YAML: {e}") from e
    if not isinstance(prices, dict):
        raise TypeError(f"Price file {path} must map model names to prices, got {type(prices).__name__}")
    for name, entry in prices.items():
        if not isinstance(name, str):
            raise TypeError(f"Price file {path} has a model name that is not text: {name!r}")
        # An entry with nothing under it is the same as a missing entry: cost unknown
        if entry is not None:
            _check_entry(path, name, entry)
    return prices


def load_prices(path=None):
    """Return the prices from prices.yaml (or path), raising an error naming the file and key if it is
    malformed. Read once, then cached."""
    return _load_file(path or PRICES_PATH)


@dataclass
class CallCost:
    """The estimated cost of one request. cost is None when it is unknown (reason says why)."""
    model: str
    tokens: dict = None
    cost: float = None
    reason: str = None
    rates: dict = None
    tier: str = None
    source: str = None


def estimate_cost(model, tokens, prices):
    """Return the CallCost of one request to model that used tokens (None: no usage reported)."""
    if tokens is None:
        return CallCost(model, reason="no token usage reported")
    entry = prices.get(model)
    if entry is None:
        return CallCost(model, tokens, reason=f"no price for {model} in {PRICES_NAME}")
    source = entry.get('source')
    rates = entry
    tier = None
    threshold = entry.get('threshold_input_tokens')
    request_input = tokens['input'] + tokens['cache_write'] + tokens['cache_read']
    if threshold is not None and request_input > threshold:
        rates = entry.get('above_threshold') or {}
        tier = f"above {threshold:,} input tokens"
    rates = {t: rates.get(t) for t in TOKEN_TYPES}
    blank = [TOKEN_LABELS[t] for t in TOKEN_TYPES if tokens[t] > 0 and rates[t] is None]
    if blank:
        tier_note = f" {tier}" if tier else ""
        return CallCost(model, tokens, reason=f"no {', '.join(blank)} price{tier_note} for {model} in {PRICES_NAME}",
                        rates=rates, tier=tier, source=source)
    cost = sum(tokens[t] * rates[t] / 1_000_000 for t in TOKEN_TYPES if tokens[t] > 0)
    return CallCost(model, tokens, cost=cost, rates=rates, tier=tier, source=source)


def format_cost(cost):
    """Return a cost in USD as e.g. "$0.0952"; a cost above 0 that would round to $0.0000 is "<$0.0001"."""
    if 0 < cost < 0.00005:
        return "<$0.0001"
    return f"${cost:,.4f}"


def format_rate(rate):
    """Return a price per 1M tokens as e.g. "$2.00", keeping any digits past the cents."""
    return f"${rate:,.2f}" if round(rate, 2) == rate else f"${rate:,}"


def _unknown_notes(costs):
    """Return the reasons the cost of some calls is unknown, each with how many calls it applies to."""
    counts = {}
    for call in costs:
        if call.cost is None:
            counts[call.reason] = counts.get(call.reason, 0) + 1
    return counts


def cost_summary(costs):
    """Return the cost part of a summary line, e.g. ", $0.0960", for the calls in costs.

    Calls that reported no usage are left out (the summary line already says so); if no call reported usage,
    the result is "". Unknown costs print as "cost unknown (<reason>)".
    """
    priced = [c for c in costs if c.tokens is not None]
    if not priced:
        return ""
    known = [c.cost for c in priced if c.cost is not None]
    unknown = _unknown_notes(priced)
    if not unknown:
        return f", {format_cost(sum(known))}"
    reasons = "; ".join(unknown)
    if not known:
        return f", cost unknown ({reasons})"
    unknown_calls = sum(unknown.values())
    return (f", {format_cost(sum(known))} + cost unknown for {unknown_calls} of {len(priced)} calls "
            f"({reasons})")


def print_cost_breakdown(costs):
    """Print the cost per token type (tokens x price) for the calls in costs, grouped by model and price
    tier. Used with --verbose."""
    groups = {}
    for call in costs:
        groups.setdefault((call.model, call.tier, call.tokens is None), []).append(call)
    for (model, tier, no_usage), calls in groups.items():
        requests = f"{len(calls)} request{'' if len(calls) == 1 else 's'}"
        if no_usage:
            print(f"  {model}: {requests} reported no token usage, not priced")
            continue
        first = calls[0]
        if first.rates is None:
            print(f"  Estimated cost (list price) for {model} ({requests}): cost unknown ({first.reason})")
            continue
        tokens = sum_tokens([c.tokens for c in calls])
        tier_note = f", {tier}" if tier else ""
        source = f", source: {first.source}" if first.source else ""
        print(f"  Estimated cost (list price) for {model} ({requests}{tier_note}{source}):")
        for t in TOKEN_TYPES:
            rate = first.rates[t]
            label = f"{TOKEN_LABELS[t]}:"
            if rate is not None:
                print(f"    {label:<12} {tokens[t]:,} tokens x {format_rate(rate)} per 1M = "
                      f"{format_cost(tokens[t] * rate / 1_000_000)}")
            elif tokens[t]:
                print(f"    {label:<12} {tokens[t]:,} tokens, no price in {PRICES_NAME}: cost unknown")
            else:
                print(f"    {label:<12} 0 tokens (no price in {PRICES_NAME}, not needed)")
        known = [c.cost for c in calls if c.cost is not None]
        if len(known) == len(calls):
            print(f"    {'total:':<12} {format_cost(sum(known))}")
        else:
            print(f"    {'total:':<12} cost unknown ({'; '.join(_unknown_notes(calls))})")


def print_cost_total(costs, verbose=False):
    """Print the run total, labeled once as an estimate at list price. Calls with an unknown cost or no
    reported usage are named, not guessed."""
    if not costs:
        return
    priced = [c for c in costs if c.tokens is not None]
    known = [c.cost for c in priced if c.cost is not None]
    unknown = _unknown_notes(priced)
    no_usage = len(costs) - len(priced)
    total = format_cost(sum(known)) if known else "unknown"
    notes = []
    if unknown and known:
        notes.append(f"plus cost unknown for {sum(unknown.values())} of {len(priced)} calls "
                     f"({'; '.join(unknown)})")
    elif unknown:
        notes.append(f"({'; '.join(unknown)})")
    if no_usage:
        notes.append(f"({no_usage} call{'' if no_usage == 1 else 's'} reported no token usage, not counted)")
    blank_line = "\n" if verbose else ""
    note = (" " + " ".join(notes)) if notes else ""
    print(f"{blank_line}Total estimated cost (list price): {total}{note}")
