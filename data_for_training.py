#!/usr/bin/env python3
"""
Pull 1-minute OHLCV for 5 years (or any duration) from Coinbase Exchange (NY-friendly, no API key).
- Products default: BTC-USD, ETH-USD, SOL-USD, DOGE-USD
- Granularity fixed to 60s (1m) for this use-case.
- Robust paging (300-candle limit), retries, exponential backoff on 429/5xx.
- Writes per-asset CSV incrementally (append mode) so you can stop/restart (resume).
- Produces a combined CSV at the end (optional, can be large).
- Validates continuity per asset and logs gaps/dupes.

Schema of output CSVs:
timestamp (UTC ISO8601), open, high, low, close, volume (base-asset units), product_id

Coinbase candle response (newest-first): [time, low, high, open, close, volume]
We normalize to ascending order and tidy columns.
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd
import requests

COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
MAX_CANDLES = 300            # per Coinbase request
GRANULARITY = 60             # 1-minute bars (fixed for this script)
MAX_RETRIES = 7
TIMEOUT = 30                 # seconds
DEFAULT_SLEEP = 0.20         # polite pause between requests (tune with --sleep)

# Map seconds to a label for filenames
GRAN_LABEL = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 21600: "6h", 86400: "1d"}[GRANULARITY]

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def iso(dt: datetime) -> str:
    # Coinbase accepts RFC3339; include timezone Z
    return dt.astimezone(timezone.utc).isoformat()

def get_with_backoff(url: str, params: dict, base_sleep: float) -> requests.Response:
    """HTTP GET with retries + exponential backoff on 429/5xx."""
    last_exc = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            # Handle rate limiting
            if r.status_code == 429:
                sleep_s = min(90, base_sleep * (2 ** i))
                log(f"429 rate limit. Sleeping {sleep_s:.2f}s ...")
                time.sleep(sleep_s)
                continue
            # Handle transient server errors
            if 500 <= r.status_code < 600:
                sleep_s = min(60, 1 + 2 * i)
                log(f"{r.status_code} server error. Sleeping {sleep_s:.2f}s ...")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r
        except (requests.RequestException, requests.Timeout) as e:
            last_exc = e
            sleep_s = min(60, base_sleep * (2 ** i))
            log(f"Request error: {e!r}. Retrying in {sleep_s:.2f}s ...")
            time.sleep(sleep_s)

    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP retries exhausted without exception context.")

def target_filename(product_id: str, outdir: str, years_back: int) -> str:
    safe = product_id.replace("-", "_")
    return os.path.join(outdir, f"{safe}_{GRAN_LABEL}_last_{years_back}y.csv")

def read_last_timestamp(path: str) -> Optional[datetime]:
    """Peek at last line of CSV to resume (timestamp ascending)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    # Read last non-empty line quickly
    with open(path, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last = f.readline().decode("utf-8", errors="ignore").strip()
    if not last:
        return None
    # CSV columns: timestamp,open,high,low,close,volume,product_id
    try:
        ts_str = last.split(",")[0]
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def write_header_if_needed(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "open", "high", "low", "close", "volume", "product_id"])

def append_rows(path: str, rows: List[Tuple[str, float, float, float, float, float, str]]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

def fetch_candles_window(product_id: str, start_dt: datetime, end_dt: datetime, base_sleep: float):
    """
    Fetch up to MAX_CANDLES candles between start_dt and end_dt (inclusive start, inclusive end).
    Coinbase returns newest-first: [time, low, high, open, close, volume] with time as unix seconds.
    """
    url = f"{COINBASE_BASE_URL}/products/{product_id}/candles"
    params = {
        "granularity": GRANULARITY,
        "start": iso(start_dt),
        "end": iso(end_dt),
    }
    r = get_with_backoff(url, params, base_sleep)
    data = r.json()
    if not isinstance(data, list):
        return []
    # Ascending by time
    data.sort(key=lambda x: x[0])
    return data

def fetch_entire_range(product_id: str, start_dt: datetime, end_dt: datetime, out_path: str, base_sleep: float):
    """
    Walk forward in windows of MAX_CANDLES * GRANULARITY seconds.
    Write to CSV incrementally (append), dedup by timestamp when resuming.
    """
    write_header_if_needed(out_path)
    # Resume logic: if file exists, move start forward to last+1min
    resume_from = read_last_timestamp(out_path)
    if resume_from is not None and resume_from > start_dt:
        start_dt = resume_from + timedelta(seconds=GRANULARITY)
        log(f"{product_id}: Resuming from {start_dt.isoformat()}")

    bucket_seconds = GRANULARITY
    step = MAX_CANDLES * bucket_seconds  # seconds per API call (for 1m: 300 minutes = 5 hours)
    cur = start_dt

    total_expected = math.ceil((end_dt - start_dt).total_seconds() / bucket_seconds)
    fetched = 0
    last_logged_pct = -1

    while cur <= end_dt:
        win_end = min(end_dt, cur + timedelta(seconds=step - 1))  # inclusive window end
        raw = fetch_candles_window(product_id, cur, win_end, base_sleep)

        # Normalize -> rows for CSV: timestamp iso, open, high, low, close, volume, product_id
        rows = []
        for item in raw:
            ts_unix, low, high, open_, close, vol = item
            ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
            rows.append([
                ts.isoformat().replace("+00:00", "Z"),
                float(open_),
                float(high),
                float(low),
                float(close),
                float(vol),
                product_id
            ])

        # Dedup with last timestamp already in file (rare, but can occur on re-runs)
        if rows:
            # If resuming and the first row <= last timestamp, trim leading overlap
            last_ts = read_last_timestamp(out_path)
            if last_ts:
                rows = [r for r in rows if datetime.fromisoformat(r[0].replace("Z", "+00:00")) > last_ts]

        append_rows(out_path, rows)

        fetched += len(rows)
        # progress
        if total_expected > 0:
            pct = int(100 * min(1.0, fetched / total_expected))
            if pct != last_logged_pct:
                log(f"{product_id}: {fetched}/{total_expected} (~{pct}%) rows written")
                last_logged_pct = pct

        # Move window
        cur = win_end + timedelta(seconds=1)

        # Be polite
        time.sleep(base_sleep)

    # Final tidy pass: sort + drop duplicates (safety)
    try:
        df = pd.read_csv(out_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df.to_csv(out_path, index=False)
        log(f"{product_id}: Final tidy complete. Total rows: {len(df)}")
    except Exception as e:
        log(f"{product_id}: Final tidy failed (non-fatal): {e!r}")

def check_continuity(path: str) -> Tuple[int, int]:
    """
    Return (gaps, duplicates) for the written CSV (1-minute continuity).
    """
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return (0, 0)
    df = df.sort_values("timestamp")
    if df.empty:
        return (0, 0)
    # Expected 60s increments
    diffs = (df["timestamp"].diff().dt.total_seconds().fillna(60)).astype(int)
    gaps = int((diffs > 60).sum())
    dups = int((diffs == 0).sum())
    return (gaps, dups)

def combine_outputs(paths: List[str], combined_path: str) -> None:
    frames = []
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                frames.append(pd.read_csv(p, parse_dates=["timestamp"]))
            except Exception as e:
                log(f"Skipping {p} due to read error: {e!r}")
    if not frames:
        log("No per-asset CSVs found to combine.")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product_id", "timestamp"])
    df.to_csv(combined_path, index=False)
    log(f"Combined CSV written: {combined_path} (rows={len(df)})")

def main():
    parser = argparse.ArgumentParser(description="Fetch 5y of 1m OHLCV from Coinbase (NY-friendly).")
    parser.add_argument("--products", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
                        help="Coinbase product ids (e.g., BTC-USD ETH-USD).")
    parser.add_argument("--years", type=int, default=5, help="Years back from now (default 5).")
    parser.add_argument("--outdir", type=str, default="./data_1m", help="Output directory for CSVs.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Base sleep between requests (seconds).")
    parser.add_argument("--no-combined", action="store_true", help="Skip writing combined CSV.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=365 * args.years)
    log(f"Fetching 1m OHLCV from {start.isoformat()} to {end.isoformat()} (≈{args.years}y)")

    written_paths = []
    for product in args.products:
        out_path = target_filename(product, args.outdir, args.years)
        log(f"=== {product}: writing to {out_path}")
        fetch_entire_range(product, start, end, out_path, base_sleep=args.sleep)
        gaps, dups = check_continuity(out_path)
        if gaps or dups:
            log(f"{product}: continuity check -> gaps={gaps}, duplicates={dups}")
        else:
            log(f"{product}: continuity check clean.")
        written_paths.append(out_path)

    if not args.no_combined:
        combined = os.path.join(args.outdir, f"crypto_usd_{GRAN_LABEL}_coinbase_last_{args.years}y.csv")
        combine_outputs(written_paths, combined)

    log("Done.")

if __name__ == "__main__":
    # Pandas low-memory CSV appends can be slow; we use csv.writer for streaming
    # and only use pandas for final tidy/combined. This keeps memory usage modest.
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
