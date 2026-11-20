import numpy as np
import pandas as pd

from src.data.features import FEATURE_COLS, build_features


def _frame(n=20, zone=1):
    return pd.DataFrame(
        {
            "PULocationID": zone,
            "date": pd.date_range("2025-01-01", periods=n),
            "trip_count": np.arange(n, dtype=float),
        }
    )


def test_no_target_leakage():
    """Changing a day's own trip_count must not change that day's features."""
    base = _frame()
    altered = base.copy()
    altered.loc[altered.index[-1], "trip_count"] = 10_000.0  # change last day's target

    f1 = build_features(base).set_index("date")
    f2 = build_features(altered).set_index("date")
    last = base["date"].iloc[-1]

    pd.testing.assert_series_equal(
        f1.loc[last, FEATURE_COLS].astype(float),
        f2.loc[last, FEATURE_COLS].astype(float),
    )


def test_rolling_features_use_only_past_days():
    d = _frame()
    f = build_features(d).sort_values("date")
    first = f.iloc[0]  # first usable day: Jan 8
    assert first["rolling_mean_7"] == d["trip_count"].iloc[0:7].mean()
    assert first["lag_1"] == d["trip_count"].iloc[6]
    assert first["lag_7"] == d["trip_count"].iloc[0]


def test_missing_days_become_zero_and_lags_are_calendar_days():
    d = _frame().drop(index=9)  # drop Jan 10
    f = build_features(d).set_index("date")
    missing = pd.Timestamp("2025-01-10")
    assert missing in f.index
    assert f.loc[missing, "trip_count"] == 0
    # lag_7 for Jan 17 must be Jan 10's value (0), not the value 7 rows back
    assert f.loc[pd.Timestamp("2025-01-17"), "lag_7"] == 0
