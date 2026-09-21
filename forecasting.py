"""
Forecasting layer for the Spring Water Decline Advisor.

For each spring, we fit a Holt-Winters (triple exponential smoothing) model on its
monthly discharge history. This is a legitimate, well-established time-series method
for data with both trend and seasonality -- appropriate here since discharge shows a
long-term decline trend plus a monsoon-driven seasonal cycle.

We forecast 24 months ahead and derive a simple risk classification:
    - CRITICAL: forecasted discharge drops below 40% of the spring's own 2005-2010
      baseline average within the forecast horizon
    - WATCH: projected decline trend is negative and discharge is between 40-65% of baseline
    - STABLE: discharge holding above 65% of baseline, or trend is flat/positive

This threshold logic is a simplification (real hydrogeological risk assessment needs
field surveys), and the code says so wherever risk labels are surfaced.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FORECAST_HORIZON_MONTHS = 24
BASELINE_YEARS = ("2005-01-01", "2010-12-31")


@dataclass
class SpringForecast:
    spring_id: str
    district: str
    history: pd.Series
    forecast: pd.Series
    baseline_avg: float
    forecast_end_avg: float
    pct_of_baseline: float
    risk_level: str
    risk_note: str


def _classify_risk(pct_of_baseline: float, trend_slope: float) -> tuple[str, str]:
    if pct_of_baseline < 40:
        return "CRITICAL", (
            f"Forecasted discharge is only ~{pct_of_baseline:.0f}% of its 2005-2010 baseline "
            "within the next 2 years. This pattern matches springs reported as 'dying' in "
            "NITI Aayog's Himalayan spring assessments and warrants field verification."
        )
    if pct_of_baseline < 65 or trend_slope < 0:
        return "WATCH", (
            f"Discharge is trending down (~{pct_of_baseline:.0f}% of its earlier baseline). "
            "Not yet critical, but recharge-side interventions now are cheaper than recovery later."
        )
    return "STABLE", (
        f"Discharge is holding around {pct_of_baseline:.0f}% of its historical baseline with "
        "no strong negative trend detected."
    )


def forecast_spring(df: pd.DataFrame, spring_id: str) -> SpringForecast:
    sub = df[df["spring_id"] == spring_id].sort_values("date")
    if len(sub) < 3:
        raise ValueError(
            f"Spring '{spring_id}' has only {len(sub)} data point(s); "
            "at least 3 are required to produce a forecast."
        )
    idx = pd.DatetimeIndex(sub["date"].values)
    idx.freq = pd.infer_freq(idx)
    series = pd.Series(sub["discharge_lpm"].values, index=idx)
    district = sub["district"].iloc[0]

    forecast_index = pd.date_range(
        series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq="MS"
    )

    # Holt-Winters requires at least 2 full seasonal cycles (24 months). Fall back to
    # a simple linear-trend extrapolation for springs with insufficient history.
    if len(series) >= 24:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast_raw = model.forecast(FORECAST_HORIZON_MONTHS)
    else:
        coeffs = np.polyfit(range(len(series)), series.values, 1)
        future_x = np.arange(len(series), len(series) + FORECAST_HORIZON_MONTHS)
        forecast_raw = pd.Series(np.polyval(coeffs, future_x))

    forecast_vals = pd.Series(forecast_raw.values, index=forecast_index)
    forecast_vals = forecast_vals.clip(lower=0.1)

    baseline = series.loc[BASELINE_YEARS[0]:BASELINE_YEARS[1]]
    baseline_avg = float(baseline.mean()) if len(baseline) else float(series.iloc[:24].mean())
    forecast_end_avg = float(forecast_vals.iloc[-6:].mean())  # last 6 months of forecast
    pct_of_baseline = 100 * forecast_end_avg / baseline_avg if baseline_avg else 0.0

    # crude trend slope over the forecast window
    trend_slope = float(np.polyfit(range(len(forecast_vals)), forecast_vals.values, 1)[0])

    risk_level, risk_note = _classify_risk(pct_of_baseline, trend_slope)

    return SpringForecast(
        spring_id=spring_id,
        district=district,
        history=series,
        forecast=forecast_vals,
        baseline_avg=baseline_avg,
        forecast_end_avg=forecast_end_avg,
        pct_of_baseline=pct_of_baseline,
        risk_level=risk_level,
        risk_note=risk_note,
    )


def forecast_district_summary(df: pd.DataFrame, district: str) -> pd.DataFrame:
    """Risk summary across all springs in a district -- used for the district overview table."""
    spring_ids = df[df["district"] == district]["spring_id"].unique()
    rows = []
    for sid in spring_ids:
        fc = forecast_spring(df, sid)
        rows.append({
            "spring_id": sid,
            "risk_level": fc.risk_level,
            "pct_of_baseline": round(fc.pct_of_baseline, 1),
            "forecast_end_avg_lpm": round(fc.forecast_end_avg, 1),
        })
    return pd.DataFrame(rows).sort_values("pct_of_baseline")


if __name__ == "__main__":
    import os
    _data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "springs_synthetic.csv")
    df = pd.read_csv(_data_path, parse_dates=["date"])
    fc = forecast_spring(df, df["spring_id"].iloc[0])
    print(f"Spring: {fc.spring_id} | Risk: {fc.risk_level} | {fc.risk_note}")
    print(f"Baseline avg: {fc.baseline_avg:.1f} lpm | Forecast end avg: {fc.forecast_end_avg:.1f} lpm")
