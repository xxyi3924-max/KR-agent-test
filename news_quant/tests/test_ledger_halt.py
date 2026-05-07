"""Ledger drawdown halt + cost-meter cap."""

from datetime import datetime, timezone

import pytest

from news_quant import cost_meter, ledger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "DB_PATH", tmp_path / "ledger.sqlite")
    monkeypatch.setattr(cost_meter, "DB_PATH", tmp_path / "costs.sqlite")
    yield


def test_drawdown_halt_fires_at_cap():
    ledger.init_capital(1000.0)
    ledger.record_equity(1100.0, "gain")  # new HWM
    ledger.record_equity(900.0, "loss")    # 18.18% drawdown from 1100
    halted = ledger.check_drawdown_halt(max_dd_pct=15.0)
    assert halted is True
    assert ledger.is_halted() is True


def test_drawdown_halt_below_cap():
    ledger.init_capital(1000.0)
    ledger.record_equity(1050.0, "gain")
    ledger.record_equity(990.0, "loss")  # ~5.7% drawdown
    halted = ledger.check_drawdown_halt(max_dd_pct=15.0)
    assert halted is False
    assert ledger.is_halted() is False


def test_halt_is_sticky_until_cleared():
    ledger.init_capital(1000.0)
    ledger.set_halt("manual")
    assert ledger.is_halted() is True
    # Even if equity recovers, halt remains until cleared
    ledger.record_equity(1500.0, "recovery")
    assert ledger.is_halted() is True
    ledger.clear_halt()
    assert ledger.is_halted() is False


def test_cost_meter_records_and_caps():
    cost_meter.record("claude-haiku-4-5", tokens_in=1_000_000, tokens_out=200_000)
    spent = cost_meter.today_spend_usd()
    # Haiku in=1.00, out=5.00 per M tokens; 1M*1 + 0.2M*5 = 1.0 + 1.0 = 2.0
    assert abs(spent - 2.0) < 1e-6


def test_cost_meter_assert_budget_raises_when_over():
    cost_meter.record("claude-sonnet-4-6", tokens_in=2_000_000, tokens_out=500_000)
    # Sonnet in=3.00, out=15.00 -> 6.0 + 7.5 = 13.5 USD
    with pytest.raises(RuntimeError):
        cost_meter.assert_budget(daily_cap_usd=10.0)
    # Under cap is fine
    cost_meter.assert_budget(daily_cap_usd=20.0)
