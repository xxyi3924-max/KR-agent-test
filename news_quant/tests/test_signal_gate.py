"""Signal-gate threshold + tradability checks."""

from news_quant.news.signal_gate import evaluate

UNIVERSE = {"AAPL", "MSFT", "NVDA"}

BASE = dict(
    threshold=0.2,
    tradable_universe=UNIVERSE,
    open_positions=set(),
    cycles_today=0,
    max_cycles_per_day=3,
    credibility=1.0,
)


def test_fires_on_strong_positive_signal():
    r = evaluate(direction=1.0, confidence=0.8, magnitude_bps=300, ticker="AAPL", **BASE)
    # score = 1 * 0.8 * 0.3 * 1.0 = 0.24 > 0.2
    assert r.fire is True
    assert r.side == "long"


def test_fires_on_strong_negative_signal_with_short_support():
    r = evaluate(direction=-1.0, confidence=0.9, magnitude_bps=400, ticker="AAPL", **BASE)
    # score = -1 * 0.9 * 0.4 * 1.0 = -0.36 → |.| > 0.2
    assert r.fire is True
    assert r.side == "short"


def test_blocks_when_already_holding():
    args = {**BASE, "open_positions": {"AAPL"}}
    r = evaluate(direction=1.0, confidence=0.8, magnitude_bps=300, ticker="AAPL", **args)
    assert r.fire is False
    assert "already" in r.reason.lower()


def test_blocks_when_cap_hit():
    args = {**BASE, "cycles_today": 3}
    r = evaluate(direction=1.0, confidence=0.8, magnitude_bps=300, ticker="MSFT", **args)
    assert r.fire is False
    assert "cap" in r.reason.lower()


def test_blocks_unknown_ticker():
    r = evaluate(direction=1.0, confidence=0.8, magnitude_bps=300, ticker="XYZQ", **BASE)
    assert r.fire is False


def test_blocks_below_threshold():
    # 1 * 0.1 * 0.3 = 0.03 < 0.2
    r = evaluate(direction=1.0, confidence=0.1, magnitude_bps=300, ticker="AAPL", **BASE)
    assert r.fire is False


def test_credibility_can_block_borderline_signal():
    # Same raw signal as test 1 but with credibility 0.5 → 0.24 * 0.5 = 0.12 < 0.2
    args = {**BASE, "credibility": 0.5}
    r = evaluate(direction=1.0, confidence=0.8, magnitude_bps=300, ticker="AAPL", **args)
    assert r.fire is False
