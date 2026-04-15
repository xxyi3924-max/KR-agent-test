"""
ETF Quant Monitor — Real-time QQQ/SPY signal-driven auto-invest.

Polls KR Broker price every ~10 seconds, aggregates into 5-minute OHLC bars,
and fires BUY/SELL market orders based on a weighted multi-signal quant system.

Signal weights (composite range −1 to +1):
  EMA crossover(9/21)   × 0.30  — trend direction
  RSI(14) normalised    × 0.25  — ETF thresholds: oversold=30, overbought=80 (neutral=55)
  Bollinger Bands(20,2) × 0.20  — mean reversion + squeeze filter
  TWAP(60-bar)          × 0.25  — price vs rolling mean

Entry : composite > 0.425  AND  annual_vol < 30%  AND  no BB squeeze (< 0.40% bw)
Exit  : composite < −0.20  AND  held ≥ 10 min  AND  net_profit > 3 bps

Fee-aware: minimum invest $100, minimum hold 10 minutes.

Persistence: _position is saved to agent_state.json so restarts don't orphan holdings.
Bar seeding: on start, KR Broker price history is used to warm up indicators.
"""

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from kr_broker import KRBrokerConnector, Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OHLC bar
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """One 1-minute OHLC bar."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Pure signal functions (stateless, no I/O)
# ---------------------------------------------------------------------------

def _ema(prices: List[float], period: int) -> float:
    """EMA using exponential smoothing over the full price list."""
    if not prices:
        return 0.0
    k = 2.0 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1.0 - k)
    return ema


def _rsi(prices: List[float], period: int = 14) -> float:
    """Simple-average RSI; returns 50 when insufficient data."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _bollinger(
    prices: List[float], period: int = 20, std_mult: float = 2.0
) -> Tuple[float, float, float, float]:
    """
    Returns (upper, mid, lower, bandwidth_pct).
    bandwidth_pct = (upper − lower) / mid × 100.
    """
    if len(prices) < period:
        mid = prices[-1] if prices else 0.0
        return mid, mid, mid, 0.0
    window = np.array(prices[-period:], dtype=float)
    mid = float(np.mean(window))
    std = float(np.std(window))
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw_pct = (upper - lower) / mid * 100.0 if mid else 0.0
    return upper, mid, lower, bw_pct


def _twap(prices: List[float], period: int = 60) -> float:
    """Rolling mean of the last `period` closes (VWAP proxy)."""
    if not prices:
        return 0.0
    return float(np.mean(prices[-period:]))


def _realized_vol_annual(prices: List[float], period: int = 60) -> float:
    """
    Annualised realised volatility computed from 1-minute bar returns.
    Trading year ≈ 252 × 390 = 98 280 minutes.
    Returns 0 when there is insufficient data.
    """
    if len(prices) < 2:
        return 0.0
    arr = np.array(prices[-period:], dtype=float)
    rets = np.diff(arr) / arr[:-1]
    return float(np.std(rets)) * np.sqrt(98_280) * 100.0


def _compute_composite(closes: List[float]) -> float:
    """
    Weighted composite signal in [−1, +1].

    EMA(9/21) cross   → normalised ±1 over ±2% spread
    RSI(14)           → symmetric around 55: RSI 30→+1, 55→0, 80→-1
    BB(20,2)          → position within band → normalised ±1
    TWAP(60)          → normalised ±1 over ±2% of TWAP
    """
    if len(closes) < 21:
        return 0.0

    price = closes[-1]
    if price == 0:
        return 0.0

    # --- EMA crossover ---
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    cross_pct = (ema9 - ema21) / price
    ema_sig = max(-1.0, min(1.0, cross_pct / 0.02))

    # --- RSI (symmetric neutral=55, matches ETF bull-trend baseline) ---
    # RSI 30 → +1.0, RSI 55 → 0.0, RSI 80 → −1.0  (slope: 1/25 per unit)
    rsi = _rsi(closes, 14)
    rsi_sig = max(-1.0, min(1.0, (55.0 - rsi) / 25.0))

    # --- Bollinger Bands ---
    upper, mid, lower, _ = _bollinger(closes, 20, 2.0)
    if upper == lower:
        bb_sig = 0.0
    elif price <= mid:
        half = mid - lower if mid != lower else 1.0
        bb_sig = min(1.0, (mid - price) / half)
    else:
        half = upper - mid if upper != mid else 1.0
        bb_sig = max(-1.0, -(price - mid) / half)

    # --- TWAP ---
    twap_val = _twap(closes, 60)
    if twap_val:
        twap_pct = (price - twap_val) / twap_val
        twap_sig = max(-1.0, min(1.0, twap_pct / 0.02))
    else:
        twap_sig = 0.0

    return ema_sig * 0.30 + rsi_sig * 0.25 + bb_sig * 0.20 + twap_sig * 0.25


# ---------------------------------------------------------------------------
# Market hours (duplicated from decision_engine to avoid circular import)
# ---------------------------------------------------------------------------

def _is_market_open() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return (14, 30) <= (now.hour, now.minute) < (21, 0)


def _minutes_to_close() -> Optional[float]:
    """Minutes remaining until market close (21:00 UTC); None if market closed."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return None
    close = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if now >= close or now.hour < 14 or (now.hour == 14 and now.minute < 30):
        return None
    return (close - now).total_seconds() / 60.0


def _minutes_since_open() -> Optional[float]:
    """Minutes elapsed since market open (14:30 UTC); None if market closed."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return None
    open_ = now.replace(hour=14, minute=30, second=0, microsecond=0)
    close = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if now < open_ or now >= close:
        return None
    return (now - open_).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Managed position
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    ticker: str
    qty: float
    entry_price: float
    entered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def minutes_held(self) -> float:
        return (datetime.now(timezone.utc) - self.entered_at).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# ETFMonitor
# ---------------------------------------------------------------------------

class ETFMonitor:
    """
    Background quant engine for QQQ / SPY auto-invest.

    Runs a single daemon thread that polls prices every `poll_interval`
    seconds, builds 1-minute bars, and executes BUY / SELL orders based
    on the composite multi-signal score.

    Thread safety: all writes to shared bar buffers and _position are
    protected by _lock.  Broker network calls are made outside the lock.

    Configuration keys (all optional):
        trading.etf_preference           default "QQQ"
        trading.etf_alternate            default "SPY"
        trading.free_cash_threshold      default 10_000
        trading.etf_monitor_min_invest   default 100
        etf_monitor.entry_threshold      default 0.55
        etf_monitor.late_entry_threshold default 0.65  (used when mins_to_close <= late_entry_minutes)
        etf_monitor.late_entry_minutes   default 120
        etf_monitor.exit_threshold       default -0.20
        etf_monitor.min_hold_minutes     default 10
        etf_monitor.min_profit_bps       default 25
        etf_monitor.max_vol_pct          default 30.0
        etf_monitor.squeeze_bw_pct       default 0.40  (5-min bar calibrated)
    """

    # Cooldown after a forced external liquidation (P4)
    _FORCED_EXIT_COOLDOWN_MINUTES = 30
    # Cooldown after a stop-loss exit — prevents re-entering a crashing market (P9)
    _STOP_LOSS_COOLDOWN_MINUTES = 60
    # Max stop-losses per calendar day before trading is suspended until next open (P9)
    _MAX_STOP_LOSSES_PER_DAY = 2

    def __init__(
        self,
        broker: KRBrokerConnector,
        config: dict,
        poll_interval: int = 10,
        state_path: str = "agent_state.json",
    ):
        self.broker = broker
        self.poll_interval = poll_interval
        self._state_path = os.path.expanduser(state_path)

        trading = config.get("trading", {})
        self.etf_primary = trading.get("etf_preference", "QQQ")
        self.etf_secondary = trading.get("etf_alternate", "SPY")
        self.free_cash_threshold = float(trading.get("free_cash_threshold", 10_000))
        self.min_invest = float(trading.get("etf_monitor_min_invest", 100))

        q = config.get("etf_monitor", {})
        self.entry_threshold = float(q.get("entry_threshold", 0.425))
        self.late_entry_threshold = float(q.get("late_entry_threshold", 0.50))
        self.late_entry_minutes = int(q.get("late_entry_minutes", 120))
        self.exit_threshold = float(q.get("exit_threshold", -0.20))
        self.min_hold_minutes = float(q.get("min_hold_minutes", 10))
        self.min_profit_bps = float(q.get("min_profit_bps", 25))
        self.max_vol_pct = float(q.get("max_vol_pct", 30.0))
        self.squeeze_bw_pct = float(q.get("squeeze_bw_pct", 0.40))
        self.stop_loss_pct = float(q.get("stop_loss_pct", 2.0))
        self.eod_close_minutes = int(q.get("eod_close_minutes", 15))
        self.no_entry_minutes = int(q.get("no_entry_minutes", 120))
        self.open_skip_minutes = int(q.get("open_skip_minutes", 30))
        self.eod_tighten_stop_minutes = int(q.get("eod_tighten_stop_minutes", 60))
        self.eod_tighten_stop_pct = float(q.get("eod_tighten_stop_pct", 1.0))
        self.breakeven_bps = float(q.get("breakeven_bps", 10.0))

        # Per-ticker bar buffers (200 bars ≈ 16.7 h of 5-min bars)
        self._bars: Dict[str, deque] = {
            self.etf_primary: deque(maxlen=200),
            self.etf_secondary: deque(maxlen=200),
        }
        # Partial bar accumulator for the current minute
        self._partial: Dict[str, Optional[dict]] = {
            self.etf_primary: None,
            self.etf_secondary: None,
        }

        self._position: Optional[_Position] = None
        self._last_sync_check: Optional[datetime] = None
        # Cooldown timestamp: block entry until this time after a forced exit (P4) or stop-loss (P9)
        self._entry_blocked_until: Optional[datetime] = None
        # Daily stop-loss circuit breaker: count and date (P9)
        self._stop_losses_today: int = 0
        self._stop_loss_date: Optional[str] = None  # YYYY-MM-DD in UTC
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            logger.debug("ETFMonitor already running")
            return
        self._stop_event.clear()
        # Warm up indicator buffers from broker history before first tick (P3)
        self._seed_bars_from_history()
        # Restore persisted position — prevents double-buy after restart (P1)
        self._restore_position()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ETFMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"ETFMonitor started: polling {self.etf_primary}/{self.etf_secondary} "
            f"every {self.poll_interval}s"
        )

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("ETFMonitor stopped")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.warning(f"ETFMonitor tick error: {e}", exc_info=True)
            self._stop_event.wait(self.poll_interval)

    def _tick(self) -> None:
        """One polling cycle: fetch prices → update bars → evaluate signals."""
        if not _is_market_open():
            return

        now = datetime.now(timezone.utc)

        # --- Step 1: fetch prices (no lock, I/O) ---
        prices: Dict[str, float] = {}
        for ticker in (self.etf_primary, self.etf_secondary):
            try:
                prices[ticker] = self.broker.get_current_price(ticker)
            except Exception as e:
                logger.debug(f"ETFMonitor: price fetch failed for {ticker}: {e}")

        if not prices:
            return

        # --- Step 2: ingest ticks, snapshot state (lock) ---
        with self._lock:
            for ticker, price in prices.items():
                self._ingest_tick(ticker, price, now)
            closes: Dict[str, List[float]] = {
                t: self._close_prices(t)
                for t in (self.etf_primary, self.etf_secondary)
            }
            position_snapshot = self._position

        # --- Step 3: compute signals (no lock, pure maths) ---
        signals: Dict[str, float] = {}
        for ticker, cls in closes.items():
            if len(cls) >= 21:
                signals[ticker] = _compute_composite(cls)

        if not signals:
            return

        # --- Step 4: decide and act (no lock during I/O) ---
        if position_snapshot is None:
            # Entry: pick best ETF
            best_ticker = max(signals, key=signals.__getitem__)
            best_signal = signals[best_ticker]
            if best_signal > self.entry_threshold:
                self._execute_entry(
                    best_ticker, best_signal, closes.get(best_ticker, [])
                )
        else:
            # Exit check: only the held ticker
            ticker = position_snapshot.ticker
            if ticker in signals:
                current_price = prices.get(ticker, closes.get(ticker, [0])[-1])
                self._execute_exit_check(
                    position_snapshot, signals[ticker], current_price
                )
            # External liquidation guard: check once per minute
            now_utc = datetime.now(timezone.utc)
            if (
                self._last_sync_check is None
                or (now_utc - self._last_sync_check).total_seconds() >= 60
            ):
                self._last_sync_check = now_utc
                self._sync_position_with_broker(position_snapshot)

    # ------------------------------------------------------------------
    # Bar management
    # ------------------------------------------------------------------

    def _ingest_tick(self, ticker: str, price: float, ts: datetime) -> None:
        """Accumulate tick into current partial bar; flush on 5-minute boundary."""
        minute_ts = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        partial = self._partial[ticker]

        if partial is None:
            self._partial[ticker] = {
                "ts": minute_ts,
                "open": price, "high": price, "low": price, "close": price,
            }
        elif partial["ts"] == minute_ts:
            partial["high"] = max(partial["high"], price)
            partial["low"] = min(partial["low"], price)
            partial["close"] = price
        else:
            # Minute rolled over — flush completed bar
            self._bars[ticker].append(Bar(
                ts=partial["ts"],
                open=partial["open"],
                high=partial["high"],
                low=partial["low"],
                close=partial["close"],
            ))
            self._partial[ticker] = {
                "ts": minute_ts,
                "open": price, "high": price, "low": price, "close": price,
            }

    def _close_prices(self, ticker: str) -> List[float]:
        """List of closed bar closes + current partial close, oldest first."""
        closes = [b.close for b in self._bars.get(ticker, [])]
        partial = self._partial.get(ticker)
        if partial:
            closes.append(partial["close"])
        return closes

    # ------------------------------------------------------------------
    # Startup: bar seeding (P3) and position restore (P1)
    # ------------------------------------------------------------------

    def _seed_bars_from_history(self) -> None:
        """
        Warm up EMA/BB/TWAP buffers from broker price history so indicators
        are meaningful immediately on first tick, not after 5 hours of
        live data. (P3 fix: 41-bar cold-start blindness)

        Broker history is tick prices (~14s intervals); we group into
        synthetic 5-min bar closes for indicator purposes only.
        """
        for ticker in (self.etf_primary, self.etf_secondary):
            try:
                history = self.broker.get_price_history(ticker, limit=200)
                if len(history) < 5:
                    logger.debug(f"ETFMonitor: insufficient history for {ticker} ({len(history)} points)")
                    continue
                # Group ticks into synthetic 1-min bars (every ~4 ticks ≈ 60s at 14s intervals)
                chunk_size = max(1, round(len(history) / min(len(history), 80)))
                dummy_ts = datetime(2000, 1, 1, tzinfo=timezone.utc)
                seeded = 0
                with self._lock:
                    for i in range(0, len(history), chunk_size):
                        chunk = history[i:i + chunk_size]
                        self._bars[ticker].append(Bar(
                            ts=dummy_ts + timedelta(minutes=seeded),
                            open=chunk[0],
                            high=max(chunk),
                            low=min(chunk),
                            close=chunk[-1],
                        ))
                        seeded += 1
                logger.info(
                    f"ETFMonitor: seeded {seeded} warmup bars for {ticker} "
                    f"from {len(history)} history points"
                )
            except Exception as e:
                logger.debug(f"ETFMonitor: bar seeding failed for {ticker}: {e}")

    def _restore_position(self) -> None:
        """
        On startup, restore _position from agent_state.json and verify with
        broker. Prevents double-buy after crashes / deploys. (P1 fix)
        """
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                state = json.load(f)
            ep = state.get("etf_position")
            if not ep:
                return
            pos = _Position(
                ticker=ep["ticker"],
                qty=float(ep["qty"]),
                entry_price=float(ep["entry_price"]),
                entered_at=datetime.fromisoformat(ep["entered_at"]),
            )
            # Verify broker still holds this position
            broker_pos = self.broker.get_position(pos.ticker)
            if broker_pos and broker_pos.qty >= pos.qty * 0.5:
                with self._lock:
                    self._position = pos
                logger.info(
                    f"ETFMonitor: restored {pos.ticker} position "
                    f"{pos.qty:.3f}sh @ ${pos.entry_price:.2f} from state"
                )
            else:
                logger.info(
                    f"ETFMonitor: stale state for {pos.ticker} "
                    f"(broker qty too low), clearing"
                )
                self._clear_etf_state()
        except Exception as e:
            logger.warning(f"ETFMonitor: position restore failed: {e}")

    # ------------------------------------------------------------------
    # State persistence helpers (P1)
    # ------------------------------------------------------------------

    def _save_etf_state(self, pos: _Position) -> None:
        """Write etf_position to agent_state.json."""
        try:
            state: dict = {}
            if os.path.exists(self._state_path):
                with open(self._state_path) as f:
                    state = json.load(f)
            state["etf_position"] = {
                "ticker": pos.ticker,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "entered_at": pos.entered_at.isoformat(),
            }
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"ETFMonitor: state save failed: {e}")

    def _clear_etf_state(self) -> None:
        """Remove etf_position from agent_state.json."""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path) as f:
                state = json.load(f)
            state.pop("etf_position", None)
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"ETFMonitor: state clear failed: {e}")

    # ------------------------------------------------------------------
    # Regime filters
    # ------------------------------------------------------------------

    def _vol_ok(self, closes: List[float]) -> bool:
        """True when realised annual vol is below the max threshold."""
        return _realized_vol_annual(closes) < self.max_vol_pct

    def _bb_squeeze(self, closes: List[float]) -> bool:
        """True when Bollinger bandwidth < squeeze threshold (low-conviction zone)."""
        _, _, _, bw_pct = _bollinger(closes, 20, 2.0)
        return bw_pct < self.squeeze_bw_pct

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def _execute_entry(
        self, ticker: str, signal: float, closes: List[float]
    ) -> None:
        """
        Place a market BUY if all entry conditions pass.
        Network calls are made before writing _position (no lock held).
        """
        # Cooldown guard: skip entry if decision_engine recently liquidated us (P4)
        if (
            self._entry_blocked_until is not None
            and datetime.now(timezone.utc) < self._entry_blocked_until
        ):
            remaining = (self._entry_blocked_until - datetime.now(timezone.utc)).total_seconds() / 60
            logger.debug(
                f"ETFMonitor: entry blocked for {remaining:.0f} more min "
                f"(post-forced-exit cooldown)"
            )
            return

        # Opening skip: no entries during first N minutes of session (price discovery / gap noise)
        mins_since_open = _minutes_since_open()
        if mins_since_open is not None and mins_since_open < self.open_skip_minutes:
            logger.debug(
                f"ETFMonitor: entry skipped — opening skip "
                f"({mins_since_open:.0f} min since open, wait {self.open_skip_minutes} min)"
            )
            return

        # No-entry window: block entries too close to close (no profit runway)
        mins_to_close = _minutes_to_close()
        if mins_to_close is not None and mins_to_close <= self.no_entry_minutes:
            logger.debug(
                f"ETFMonitor: entry skipped — no-entry window "
                f"({mins_to_close:.0f} min to close, cutoff={self.no_entry_minutes} min)"
            )
            return

        # Late-session gate: require higher conviction when runway is short
        effective_threshold = (
            self.late_entry_threshold
            if mins_to_close is not None and mins_to_close <= self.late_entry_minutes
            else self.entry_threshold
        )
        if signal <= effective_threshold:
            logger.debug(
                f"ETFMonitor: entry skipped for {ticker} — signal {signal:.3f} ≤ "
                f"{'late' if mins_to_close is not None and mins_to_close <= self.late_entry_minutes else 'normal'} "
                f"threshold {effective_threshold:.2f}"
            )
            return

        # Regime filters
        if closes and not self._vol_ok(closes):
            logger.debug(
                f"ETFMonitor: entry skipped for {ticker} — "
                f"annual vol {_realized_vol_annual(closes):.1f}% ≥ {self.max_vol_pct}%"
            )
            return
        if closes and self._bb_squeeze(closes):
            logger.debug(f"ETFMonitor: entry skipped for {ticker} — BB squeeze")
            return

        # Fetch current cash (I/O, no lock)
        try:
            account = self.broker.get_account_info()
        except Exception as e:
            logger.warning(f"ETFMonitor: account fetch failed during entry: {e}")
            return

        invest_amount = account.cash - self.free_cash_threshold
        if invest_amount < self.min_invest:
            logger.debug(
                f"ETFMonitor: entry skipped — invest amount "
                f"${invest_amount:.2f} < min ${self.min_invest}"
            )
            return

        price = closes[-1] if closes else 0.0
        if price <= 0:
            return

        # Place order (I/O, no lock)
        try:
            order = Order(
                ticker=ticker,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                qty=invest_amount / price,
                price=price,
                dollar_amount=invest_amount,
            )
            self.broker.place_order(order)
        except Exception as e:
            logger.error(f"ETFMonitor: BUY order failed for {ticker}: {e}")
            return

        # Update state (lock, TOCTOU check)
        with self._lock:
            if self._position is None:
                self._position = _Position(
                    ticker=ticker,
                    qty=invest_amount / price,
                    entry_price=price,
                )
                logger.info(
                    f"ETFMonitor BUY  {ticker}: ${invest_amount:.0f} "
                    f"({invest_amount/price:.3f} shares @ ${price:.2f}) "
                    f"signal={signal:.3f}"
                )
                # Persist so a restart doesn't re-buy (P1)
                self._save_etf_state(self._position)
            else:
                logger.warning(
                    f"ETFMonitor: position already opened by another tick, "
                    f"skipping {ticker} entry record (order already placed)"
                )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _execute_exit_check(
        self,
        pos: _Position,
        signal: float,
        current_price: float,
    ) -> None:
        """
        Evaluate exit conditions and place SELL if any pass:
          Normal exit : composite < exit_threshold AND held ≥ min_hold AND profit > 3 bps
          Stop-loss   : price dropped > stop_loss_pct% below entry (overrides profit gate)
          EOD close   : < eod_close_minutes before market close (overrides all gates)
        """
        if pos.minutes_held() < self.min_hold_minutes:
            return

        # --- Session-phase stop-loss calculation ---
        mins_left = _minutes_to_close()
        near_close = mins_left is not None and mins_left <= self.eod_tighten_stop_minutes

        if near_close:
            # Tighter stop in final N minutes: less time to recover from adverse move
            stop_pct = self.eod_tighten_stop_pct
        else:
            stop_pct = self.stop_loss_pct
        stop_price = pos.entry_price * (1.0 - stop_pct / 100.0)

        # --- Breakeven stop: if profitable near close, floor stop at entry ---
        current_profit_bps = (current_price - pos.entry_price) / pos.entry_price * 10_000.0
        if near_close and current_profit_bps >= self.breakeven_bps:
            stop_price = max(stop_price, pos.entry_price)

        # --- EOD forced close ---
        force_eod = mins_left is not None and mins_left <= self.eod_close_minutes
        force_stop = current_price <= stop_price

        if not force_eod and not force_stop:
            # Normal exit: need signal AND profit gate
            if signal >= self.exit_threshold:
                return
            min_price = pos.entry_price * (1.0 + self.min_profit_bps / 10_000.0)
            if current_price <= min_price:
                logger.debug(
                    f"ETFMonitor: exit skipped for {pos.ticker} — "
                    f"price ${current_price:.2f} ≤ min-profit ${min_price:.4f}"
                )
                return

        if force_eod:
            reason = f"EOD/MOC ({mins_left:.0f} min to close)"
        elif force_stop:
            if near_close:
                reason = f"EOD-stop ({stop_pct}% tightened, ${current_price:.2f} ≤ ${stop_price:.2f})"
            else:
                reason = f"stop-loss (${current_price:.2f} ≤ ${stop_price:.2f})"
        else:
            reason = f"signal (comp={signal:.3f})"

        # Place sell order (I/O, no lock)
        # Fetch actual broker qty so floating-point qty estimate never exceeds holdings.
        try:
            broker_pos = self.broker.get_position(pos.ticker)
            sell_qty = broker_pos.qty if broker_pos else pos.qty
            order = Order(
                ticker=pos.ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=sell_qty,
                price=current_price,
            )
            result = self.broker.place_order(order)
        except Exception as e:
            logger.error(f"ETFMonitor: SELL order failed for {pos.ticker}: {e}")
            return

        if result.get("filledCount", 0) == 0:
            failures = result.get("failures", result.get("errors", []))
            logger.error(
                f"ETFMonitor: SELL {pos.ticker} rejected by broker — "
                f"position retained for retry. failures={failures}"
            )
            return

        profit_bps = (current_price - pos.entry_price) / pos.entry_price * 10_000.0
        logger.info(
            f"ETFMonitor SELL {pos.ticker}: {pos.qty:.3f} shares @ ${current_price:.2f} "
            f"(entry ${pos.entry_price:.2f}, P&L={profit_bps:+.1f} bps, "
            f"held={pos.minutes_held():.1f} min, reason={reason})"
        )

        with self._lock:
            if self._position and self._position.ticker == pos.ticker:
                self._position = None
        self._clear_etf_state()
        # After a stop-loss: cooldown + daily circuit breaker (P9)
        if force_stop:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._stop_loss_date != today:
                self._stop_loss_date = today
                self._stop_losses_today = 0
            self._stop_losses_today += 1

            if self._stop_losses_today >= self._MAX_STOP_LOSSES_PER_DAY:
                # Suspend trading for the rest of the day
                next_open = datetime.now(timezone.utc).replace(
                    hour=14, minute=30, second=0, microsecond=0
                ) + timedelta(days=1)
                # Skip to next weekday
                while next_open.weekday() >= 5:
                    next_open += timedelta(days=1)
                self._entry_blocked_until = next_open
                logger.warning(
                    f"ETFMonitor: {self._stop_losses_today} stop-losses today — "
                    f"circuit breaker: no entries until {next_open.strftime('%Y-%m-%d %H:%M')} UTC"
                )
            else:
                self._entry_blocked_until = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=self._STOP_LOSS_COOLDOWN_MINUTES)
                )
                logger.info(
                    f"ETFMonitor: stop-loss #{self._stop_losses_today} today — "
                    f"cooldown {self._STOP_LOSS_COOLDOWN_MINUTES} min "
                    f"(no re-entry until {self._entry_blocked_until.strftime('%H:%M')} UTC)"
                )

    # ------------------------------------------------------------------
    # External liquidation sync
    # ------------------------------------------------------------------

    def _sync_position_with_broker(self, pos: _Position) -> None:
        """
        Clear the tracked position if decision_engine liquidated it externally
        (e.g., to fund a TDash signal).  Uses a lightweight position check.

        Sets a 30-minute cooldown before ETFMonitor can re-enter, to avoid
        immediately fighting the decision_engine's direction. (P4 fix)
        """
        try:
            broker_pos = self.broker.get_position(pos.ticker)
            if broker_pos is None or broker_pos.qty < pos.qty * 0.10:
                logger.info(
                    f"ETFMonitor: {pos.ticker} position liquidated externally "
                    f"— clearing tracked position, cooling down "
                    f"{self._FORCED_EXIT_COOLDOWN_MINUTES} min"
                )
                with self._lock:
                    if self._position and self._position.ticker == pos.ticker:
                        self._position = None
                self._clear_etf_state()
                # Block re-entry for cooldown period (P4)
                self._entry_blocked_until = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=self._FORCED_EXIT_COOLDOWN_MINUTES)
                )
        except Exception:
            pass  # On error, leave state unchanged

    # ------------------------------------------------------------------
    # Debug / status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a human-readable status snapshot (safe to call anytime)."""
        with self._lock:
            pos = self._position
            bar_counts = {t: len(self._bars[t]) for t in self._bars}
        blocked_min = None
        if self._entry_blocked_until and datetime.now(timezone.utc) < self._entry_blocked_until:
            blocked_min = round(
                (self._entry_blocked_until - datetime.now(timezone.utc)).total_seconds() / 60, 1
            )
        return {
            "running": self.is_running(),
            "bars": bar_counts,
            "entry_blocked_min": blocked_min,
            "position": {
                "ticker": pos.ticker,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "minutes_held": round(pos.minutes_held(), 1),
            } if pos else None,
        }
