"""LLM scoring of 8-K filings.

For each 8-K, fetch the primary document (HTML), strip tags, send to Haiku
with a structured-output prompt.  Output JSON:

  {direction: -1.0..+1.0,
   magnitude_bps: 0..1000,
   horizon_min: int,
   confidence: 0..1,
   key_phrase: str,
   reasoning: str (short)}

Direction: +1 strongly bullish for the issuer, -1 strongly bearish, 0 neutral.
Magnitude: estimated absolute price move over horizon, in bps.
Confidence: how sure the model is about direction.

The triage step (is_actionable + ticker resolution) is skipped here because
the 8-K backfill is already pre-filtered: it's all S&P 500 issuers, and the
filing's CIK→ticker mapping is already known.  Triage is for Phase 4 web
news where 80% of articles are noise.
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from anthropic import Anthropic

from news_quant import cost_meter
from news_quant.config_loader import load as load_config

DATA_DIR = Path(__file__).parent.parent / "data"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        s = data.strip()
        if s:
            self.parts.append(s)


def html_to_text(html: str, max_chars: int = 8000) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return html[:max_chars]
    text = " ".join(p.parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def fetch_primary_text(url: str, user_agent: str, session: requests.Session, max_chars: int = 8000) -> str:
    if not url:
        return ""
    try:
        r = session.get(url, headers={"User-Agent": user_agent}, timeout=30)
        if r.status_code != 200:
            return ""
        ctype = r.headers.get("Content-Type", "")
        body = r.text
        if "html" in ctype.lower() or "<html" in body[:200].lower():
            return html_to_text(body, max_chars=max_chars)
        return body[:max_chars]
    except Exception:
        return ""


SCORING_SYSTEM = """You are a quantitative news analyst scoring SEC 8-K filings for short-term equity-price impact.

Your job: read the filing snippet and the Item codes, then estimate how the issuer's stock price will move in the 1-4 hours after the filing becomes public.

Return ONLY a JSON object with these fields, no other text:
- direction: float in [-1.0, 1.0]. +1 = strongly bullish for issuer, -1 = strongly bearish, 0 = neutral.
- magnitude_bps: integer 0..1000. Absolute expected price move in bps over horizon.
- horizon_min: integer. Minutes until move plays out (typical: 60-240).
- confidence: float in [0.0, 1.0]. How certain you are about direction. Most filings should be 0.3-0.6; reserve >0.8 for unambiguous signals (huge earnings beat/miss, major M&A, FDA approval).
- key_phrase: string <=80 chars. The single most important phrase from the filing.
- reasoning: string <=200 chars. Brief justification.

Item code reference:
  1.01 material agreement, 1.02 termination, 2.01 acquisition completion, 2.02 results of operations (earnings),
  2.03 financial obligation, 2.05 cost-cut/restructure, 3.01 delisting, 3.03 modification of rights,
  4.01 auditor change, 4.02 non-reliance on financials, 5.02 officer departure/election, 5.07 shareholder vote,
  7.01 Reg FD, 8.01 other events, 9.01 financial statements (always paired with another item).

Earnings (2.02): direction depends on beat vs guidance — if numbers reported beat consensus, bullish; if miss, bearish.
Officer departures (5.02) without explanation: usually mildly bearish unless succession is clearly orderly.
Material agreements (1.01): can be bullish (new contract) or bearish (debt issuance) — read the snippet.
"""


SCORING_USER_TEMPLATE = """Issuer: {issuer_name} ({ticker})
Filed: {acceptance_dt_utc}
Item codes: {items}

Filing snippet (first 6000 chars):
{snippet}
{extra_context_block}
Return JSON only."""


def score_filing(
    client: Anthropic,
    model: str,
    issuer_name: str,
    ticker: str,
    acceptance_dt: str,
    items: str,
    snippet: str,
    cost_tag: str = "score",
    extra_context: str = "",
) -> Optional[dict]:
    extra_block = f"\n{extra_context}\n" if extra_context else ""
    user = SCORING_USER_TEMPLATE.format(
        issuer_name=issuer_name or "(unknown)",
        ticker=ticker or "(unknown)",
        acceptance_dt_utc=acceptance_dt,
        items=items or "(none)",
        snippet=snippet[:6000] or "(empty)",
        extra_context_block=extra_block,
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=SCORING_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # Fail fast on systemic errors that will affect every subsequent call.
        for fatal in ("credit balance", "invalid_api_key", "authentication_error",
                      "permission_error", "Unauthorized"):
            if fatal in str(e):
                raise RuntimeError(f"FATAL (halting): {msg}") from e
        return {"error": msg}

    usage = resp.usage
    cost_meter.record(model, usage.input_tokens, usage.output_tokens, tag=cost_tag)

    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    # Strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a {...} substring
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {"error": "no_json", "raw": text[:500]}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"error": "json_parse_fail", "raw": text[:500]}

    obj["_tokens_in"] = usage.input_tokens
    obj["_tokens_out"] = usage.output_tokens
    return obj


def _filter_universe(
    df: pd.DataFrame,
    exclude_item_prefixes: Optional[list[str]] = None,
    exclude_accessions: Optional[set[str]] = None,
) -> pd.DataFrame:
    out = df
    if exclude_item_prefixes:
        items = out["items"].fillna("")
        mask = pd.Series(False, index=out.index)
        for pref in exclude_item_prefixes:
            mask = mask | items.str.startswith(pref)
        out = out[~mask]
    if exclude_accessions:
        out = out[~out["accession"].isin(exclude_accessions)]
    return out.reset_index(drop=True)


def score_dataframe(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    seed: int = 42,
    out_path: Optional[Path] = None,
    daily_cap_usd: float = 10.0,
    exclude_item_prefixes: Optional[list[str]] = None,
    exclude_accessions: Optional[set[str]] = None,
) -> pd.DataFrame:
    cfg = load_config()
    ua = cfg["http"]["user_agent"]
    api_key = cfg["llm"]["api_key"]
    if not api_key:
        raise RuntimeError(f"ANTHROPIC_API_KEY env var is empty.")
    model = cfg["llm"]["score_model"]

    if exclude_item_prefixes or exclude_accessions:
        before = len(df)
        df = _filter_universe(df, exclude_item_prefixes, exclude_accessions)
        print(
            f"  filtered universe: {before} → {len(df)} "
            f"(excluded items={exclude_item_prefixes or []}, "
            f"excluded accessions={len(exclude_accessions or [])})"
        )

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    client = Anthropic(api_key=api_key)
    session = requests.Session()
    rows: list[dict] = []
    t0 = time.time()
    for i, row in df.iterrows():
        cost_meter.assert_budget(daily_cap_usd)
        snippet = fetch_primary_text(row["primary_url"], ua, session, max_chars=8000)
        # SEC fair-access politeness: 100ms between fetches
        time.sleep(0.1)

        scored = score_filing(
            client,
            model,
            row.get("issuer_name", ""),
            row.get("ticker", ""),
            str(row.get("acceptance_dt_utc", "")),
            row.get("items", ""),
            snippet,
            extra_context="",
        )
        if scored is None:
            continue
        out_row = dict(row)
        out_row.update(scored)
        out_row["snippet_len"] = len(snippet)
        rows.append(out_row)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            spent = cost_meter.today_spend_usd()
            print(
                f"  scored {i+1}/{len(df)}  rate={rate:.2f}/s  "
                f"elapsed={elapsed:.0f}s  spent_today=${spent:.4f}"
            )

    out = pd.DataFrame(rows)
    if out_path:
        out.to_parquet(out_path, index=False)
    return out


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="parquet from edgar_8k.py")
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--cap-usd", type=float, default=5.0)
    p.add_argument(
        "--exclude-items",
        default="",
        help="comma-separated item-code prefixes to drop (e.g. '2.02')",
    )
    p.add_argument(
        "--exclude-accessions-from",
        default=None,
        help="parquet whose 'accession' values should be dropped from the universe",
    )
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    print(f"loaded {len(df)} events from {args.input}")
    print(f"sampling {args.sample} (seed={args.seed}); LLM cap=${args.cap_usd}")

    excl_items = [s.strip() for s in args.exclude_items.split(",") if s.strip()]
    excl_acc: Optional[set[str]] = None
    if args.exclude_accessions_from:
        prev = pd.read_parquet(args.exclude_accessions_from)
        excl_acc = set(prev["accession"].dropna().tolist())
        print(f"  will exclude {len(excl_acc)} accessions from {args.exclude_accessions_from}")

    out = (
        Path(args.out)
        if args.out
        else DATA_DIR / f"scored_sample{args.sample}_seed{args.seed}.parquet"
    )
    scored = score_dataframe(
        df,
        sample_n=args.sample,
        seed=args.seed,
        out_path=out,
        daily_cap_usd=args.cap_usd,
        exclude_item_prefixes=excl_items or None,
        exclude_accessions=excl_acc,
    )
    print(f"\nWrote {len(scored)} scored → {out}")

    print(f"\nTotal LLM spend today: ${cost_meter.today_spend_usd():.4f}")
    if "error" in scored.columns:
        errs = scored[scored["error"].notna()]
        print(f"errors: {len(errs)}/{len(scored)}")
    if "direction" in scored.columns:
        print(f"\ndirection distribution:\n{scored['direction'].describe()}")
        print(f"\nconfidence distribution:\n{scored['confidence'].describe()}")


if __name__ == "__main__":
    main()
