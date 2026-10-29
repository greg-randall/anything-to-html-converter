"""Cost estimates from token counts and prices.yaml (no API calls)."""
import re

import pytest

PRICES = {
    'sonnet': {'source': 'test', 'input': 2.00, 'output': 10.00, 'cache_write': 2.50, 'cache_read': 0.20},
    'haiku': {'source': 'test', 'input': 0.10, 'output': 0.50, 'cache_write': None, 'cache_read': None,
              'threshold_input_tokens': 100000,
              'above_threshold': {'input': 0.50, 'output': 2.50, 'cache_write': None, 'cache_read': None}},
    'blank': {'source': 'not filled in', 'input': None, 'output': None},
    'listed-without-prices': None,
    'no-above-prices': {'input': 1.0, 'output': 1.0, 'threshold_input_tokens': 10},
}


def tokens(api, **counts):
    return api.make_tokens(**counts)


def test_project_prices_file(api):
    prices = api.load_prices()
    assert {name: (e['input'], e['output'], e['cache_write'], e['cache_read']) for name, e in prices.items()} == {
        'claude-sonnet-5-5': (2.00, 10.00, None, None),
        'claude-haiku-5-5': (0.10, 0.50, None, None),
        'claude-opus-5-5': (4.00, 20.00, None, None),
        'gpt-6.1-terra': (1.00, 5.00, None, 0.05),
    }
    assert prices['claude-haiku-5-5']['threshold_input_tokens'] == 100000
    assert prices['claude-haiku-5-5']['above_threshold'] == {'input': 0.50, 'output': 2.50, 'cache_write': None,
                                                             'cache_read': None}
    assert all(entry['source'] for entry in prices.values())


def test_input_and_output_math(api):
    cost = api.estimate_cost('sonnet', tokens(api, input=8061, output=7987), PRICES)
    assert cost.cost == pytest.approx(8061 * 2.00 / 1e6 + 7987 * 10.00 / 1e6)
    assert (cost.reason, cost.tier) == (None, None)
    assert api.format_cost(cost.cost) == "$0.0960"


def test_cache_tokens_are_priced_at_their_own_rates(api):
    cost = api.estimate_cost('sonnet', tokens(api, input=100, output=50, cache_write=4000, cache_read=20000), PRICES)
    assert cost.cost == pytest.approx((100 * 2.00 + 50 * 10.00 + 4000 * 2.50 + 20000 * 0.20) / 1e6)


def test_threshold_uses_the_higher_rate_only_above_it(api):
    at = api.estimate_cost('haiku', tokens(api, input=100000, output=1000), PRICES)
    assert at.tier is None
    assert at.cost == pytest.approx(100000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)
    above = api.estimate_cost('haiku', tokens(api, input=100001, output=1000), PRICES)
    assert above.tier == "above 100,000 input tokens"
    # The higher rate applies to the whole request, output included
    assert above.cost == pytest.approx(100001 * 0.50 / 1e6 + 1000 * 2.50 / 1e6)


def test_threshold_counts_cache_tokens_as_input(api):
    # 60,000 uncached + 50,000 cache read is 110,000 input tokens: above the threshold. The above-threshold cache
    # read price is blank, so the cost is unknown rather than guessed.
    cost = api.estimate_cost('haiku', tokens(api, input=60000, output=10, cache_read=50000), PRICES)
    assert cost.cost is None
    assert cost.reason == "no cache read price above 100,000 input tokens for haiku in prices.yaml"


def test_threshold_without_above_threshold_prices_is_unknown_above_it(api):
    assert api.estimate_cost('no-above-prices', tokens(api, input=5, output=5), PRICES).cost == pytest.approx(1e-5)
    cost = api.estimate_cost('no-above-prices', tokens(api, input=11, output=5), PRICES)
    assert cost.reason == "no input, output price above 10 input tokens for no-above-prices in prices.yaml"


def test_missing_entry_or_entry_without_prices(api):
    for model in ('not-listed', 'listed-without-prices'):
        cost = api.estimate_cost(model, tokens(api, input=10, output=10), PRICES)
        assert (cost.cost, cost.reason) == (None, f"no price for {model} in prices.yaml")
        assert api.cost_summary([cost]) == f", cost unknown (no price for {model} in prices.yaml)"


def test_blank_price_is_unknown_only_when_needed(api):
    cost = api.estimate_cost('blank', tokens(api, input=10, output=10), PRICES)
    assert (cost.cost, cost.reason) == (None, "no input, output price for blank in prices.yaml")
    # Blank cache prices don't matter when the request used no cache tokens
    assert api.estimate_cost('haiku', tokens(api, input=10, output=10), PRICES).cost == pytest.approx(6e-6)
    cost = api.estimate_cost('haiku', tokens(api, input=10, output=10, cache_write=5), PRICES)
    assert cost.reason == "no cache write price for haiku in prices.yaml"


def test_no_usage_is_not_priced(api):
    cost = api.estimate_cost('sonnet', None, PRICES)
    assert (cost.tokens, cost.cost, cost.reason) == (None, None, "no token usage reported")
    assert api.cost_summary([cost]) == ""


def test_format_cost(api):
    assert api.format_cost(0) == "$0.0000"
    assert api.format_cost(0.0000205) == "<$0.0001"
    assert api.format_cost(0.09520) == "$0.0952"
    assert api.format_cost(1234.5) == "$1,234.5000"


def test_cost_summary_sums_requests_and_names_unknown_ones(api):
    known = [api.estimate_cost('sonnet', tokens(api, input=1000, output=100), PRICES) for _ in range(3)]
    assert api.cost_summary(known) == ", $0.0090"
    unknown = api.estimate_cost('blank', tokens(api, input=1, output=1), PRICES)
    no_usage = api.estimate_cost('sonnet', None, PRICES)
    assert api.cost_summary(known + [unknown, no_usage]) == (
        ", $0.0090 + cost unknown for 1 of 4 calls (no input, output price for blank in prices.yaml)")


def test_total_line(api, capsys):
    known = api.estimate_cost('sonnet', tokens(api, input=8061, output=7987), PRICES)
    unknown = api.estimate_cost('not-listed', tokens(api, input=1, output=1), PRICES)
    no_usage = api.estimate_cost('sonnet', None, PRICES)
    api.print_cost_total([known])
    api.print_cost_total([known, unknown, no_usage])
    api.print_cost_total([unknown])
    api.print_cost_total([known], verbose=True)
    api.print_cost_total([])
    assert capsys.readouterr().out == (
        "Total estimated cost (list price): $0.0960\n"
        "Total estimated cost (list price): $0.0960 plus cost unknown for 1 of 2 calls (no price for not-listed "
        "in prices.yaml) (1 call reported no token usage, not counted)\n"
        "Total estimated cost (list price): unknown (no price for not-listed in prices.yaml)\n"
        "\nTotal estimated cost (list price): $0.0960\n")


def test_verbose_breakdown(api, capsys):
    costs = [api.estimate_cost('haiku', tokens(api, input=1000, output=20), PRICES),
             api.estimate_cost('haiku', tokens(api, input=1000, output=20), PRICES),
             api.estimate_cost('haiku', tokens(api, input=200000, output=20), PRICES),
             api.estimate_cost('haiku', None, PRICES),
             api.estimate_cost('blank', tokens(api, input=5, output=0), PRICES),
             api.estimate_cost('not-listed', tokens(api, input=5, output=5), PRICES)]
    api.print_cost_breakdown(costs)
    assert capsys.readouterr().out == (
        "  Estimated cost (list price) for haiku (2 requests, source: test):\n"
        "    input:       2,000 tokens x $0.10 per 1M = $0.0002\n"
        "    output:      40 tokens x $0.50 per 1M = <$0.0001\n"
        "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
        "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
        "    total:       $0.0002\n"
        "  Estimated cost (list price) for haiku (1 request, above 100,000 input tokens, source: test):\n"
        "    input:       200,000 tokens x $0.50 per 1M = $0.1000\n"
        "    output:      20 tokens x $2.50 per 1M = $0.0001\n"
        "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
        "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
        "    total:       $0.1001\n"
        "  haiku: 1 request reported no token usage, not priced\n"
        "  Estimated cost (list price) for blank (1 request, source: not filled in):\n"
        "    input:       5 tokens, no price in prices.yaml: cost unknown\n"
        "    output:      0 tokens (no price in prices.yaml, not needed)\n"
        "    cache write: 0 tokens (no price in prices.yaml, not needed)\n"
        "    cache read:  0 tokens (no price in prices.yaml, not needed)\n"
        "    total:       cost unknown (no input price for blank in prices.yaml)\n"
        "  Estimated cost (list price) for not-listed (1 request): cost unknown (no price for not-listed in "
        "prices.yaml)\n")


@pytest.mark.parametrize('text, error, message', [
    ("a: [1, 2\n", ValueError, "is not valid YAML"),
    ("- a\n- b\n", TypeError, "must map model names to prices, got list"),
    ("", TypeError, "must map model names to prices, got NoneType"),
    ("m:\n  input: 1\n  ouput: 2\n", KeyError, "Entry 'm' in {path} has an unknown key 'ouput'"),
    ("m:\n  input: two\n", TypeError, "Entry 'm', key 'input' in {path} must be a price"),
    ("m:\n  input: -1\n", TypeError, "Entry 'm', key 'input' in {path} must be a price"),
    ("m:\n  input: true\n", TypeError, "Entry 'm', key 'input' in {path} must be a price"),
    ("m: 5\n", TypeError, "Entry 'm' in {path} must be a mapping"),
    ("m:\n  source: 5\n", TypeError, "Entry 'm', key 'source' in {path} must be text"),
    ("m:\n  threshold_input_tokens: 0\n", TypeError, "key 'threshold_input_tokens' in {path} must be a whole"),
    ("m:\n  above_threshold:\n    input: 1\n", KeyError, "Entry 'm' in {path} has 'above_threshold' prices but no"),
    ("m:\n  threshold_input_tokens: 5\n  above_threshold:\n    input: x\n", TypeError,
     "Entry 'm', key 'above_threshold', key 'input' in {path} must be a price"),
    ("m:\n  threshold_input_tokens: 5\n  above_threshold:\n    source: x\n", KeyError,
     "Entry 'm', key 'above_threshold' in {path} has an unknown key 'source'"),
])
def test_malformed_price_file_errors_name_the_file_and_key(api, tmp_path, text, error, message):
    path = tmp_path / 'prices.yaml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(error, match=re.escape(message.format(path=path))):
        api.load_prices(str(path))


def test_missing_price_file_is_an_error(api, tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape(f"Price file {tmp_path / 'none.yaml'} is missing")):
        api.load_prices(str(tmp_path / 'none.yaml'))
