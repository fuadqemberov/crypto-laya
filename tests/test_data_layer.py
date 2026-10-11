"""Parsing, UTC handling, de-duplication, gaps and quality checks."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from conftest import T0, synthetic_klines
from laya_crypto.ingestion.parsers import KLINE_COLS, normalise, parse_funding, parse_klines, parse_metrics
from laya_crypto.utils.timeframes import horizon_bars, tf_ms, to_utc_ms
from laya_crypto.validation.quality import find_gaps, kline_quality, ohlc_violations


def _zip(tmp_path, name, csv_text):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), csv_text)
    return p


def _kline_csv(df, header):
    buf = io.StringIO()
    out = df.assign(ignore=0)[KLINE_COLS]
    out.to_csv(buf, index=False, header=header)
    return buf.getvalue()


def test_to_utc_ms_detects_units():
    ms = 1_735_689_600_000
    got = to_utc_ms([ms // 1000, ms, ms * 1000, ms * 1_000_000])
    assert (got == ms).all()


def test_timeframe_helpers():
    assert tf_ms("5m") == 300_000 and tf_ms("4h") == 14_400_000
    assert horizon_bars(60, "5m") == 12


def test_parse_klines_with_and_without_header(tmp_path):
    df = synthetic_klines(50)
    a = parse_klines(_zip(tmp_path, "a.zip", _kline_csv(df, True)))
    b = parse_klines(_zip(tmp_path, "b.zip", _kline_csv(df, False)))
    pd.testing.assert_frame_equal(a, b)
    assert a["open_time"].iloc[0] == T0
    assert pd.to_datetime(a["open_time"].iloc[0], unit="ms", utc=True) == pd.Timestamp("2025-01-01", tz="UTC")


def test_parse_klines_microseconds(tmp_path):
    df = synthetic_klines(5)
    df["open_time"] *= 1000
    df["close_time"] *= 1000
    out = parse_klines(_zip(tmp_path, "us.zip", _kline_csv(df, True)))
    assert out["open_time"].iloc[0] == T0


def test_normalise_drops_duplicates_and_sorts():
    df = synthetic_klines(10)
    dup = pd.concat([df.iloc[[3]], df, df.iloc[[5]]]).iloc[::-1]
    out, dropped = normalise(dup, "klines")
    assert dropped == 2 and out["open_time"].is_monotonic_increasing and len(out) == 10


def test_parse_funding_and_metrics(tmp_path):
    f = parse_funding(_zip(tmp_path, "f.zip", "calc_time,funding_interval_hours,last_funding_rate\n"
                                              "1754006400001,8,-0.00001408\n"))
    assert f["funding_rate"].iloc[0] == -0.00001408 and f["funding_time"].iloc[0] == 1754006400001
    m = parse_metrics(_zip(tmp_path, "m.zip",
                           "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
                           "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
                           "2025-08-01 00:05:00,BTCUSDT,90260.4,10440856136.1,1.27,1.94,1.16,1.88\n"))
    assert m["time"].iloc[0] == pd.Timestamp("2025-08-01 00:05", tz="UTC").value // 1_000_000


def test_gap_detection_and_quality():
    df = synthetic_klines(100)
    holed = df.drop(index=range(40, 45)).reset_index(drop=True)
    gaps = find_gaps(holed["open_time"], "5m")
    assert len(gaps) == 1 and gaps["missing_bars"].iloc[0] == 5
    q = kline_quality(holed, "5m", "X")
    assert q["missing_bars"] == 5 and q["quality_score"] < 1.0
    assert kline_quality(df, "5m", "X")["quality_score"] > q["quality_score"]


def test_ohlc_violation_detected():
    df = synthetic_klines(10)
    df.loc[3, "high"] = df.loc[3, "low"] - 1
    assert ohlc_violations(df).sum() == 1 and not np.isnan(df["high"]).any()
