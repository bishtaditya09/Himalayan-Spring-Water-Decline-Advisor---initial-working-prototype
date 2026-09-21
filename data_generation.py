"""
Synthetic data generator for the Himalayan Spring Water Decline Advisor.

IMPORTANT: This data is SYNTHETIC. Village-level spring discharge records are not
publicly available at scale, so we simulate plausible discharge time series that are
grounded in real, documented trends reported by NITI Aayog's 2018 report on Himalayan
springs and district-level rainfall patterns for Uttarakhand. This lets the forecasting
and risk-classification logic be demonstrated meaningfully, while being transparent
that it is not measured field data.

Grounding assumptions (documented, not invented):
- Many Himalayan springs show a declining discharge trend over the last two decades,
  frequently cited in the 20-50% range depending on region and land-use change.
- Discharge is seasonal: peaks post-monsoon (Aug-Oct), troughs in pre-monsoon
  (Apr-Jun), consistent with recharge-dependent hill aquifers.
- Springs in districts with higher deforestation / construction activity or lower
  rainfall trend decline faster than those in better-forested, higher-rainfall areas.
- Rainfall itself is simulated with a monsoon-dominated seasonal pattern plus
  year-to-year variability, not a flat trend.

This module is deliberately kept simple and readable so the assumptions above are
easy to audit and adjust.
"""

import os

import numpy as np
import pandas as pd

RNG_SEED = 42

# Representative hill districts of Uttarakhand with rough characteristics used to
# vary the simulation (not precise figures -- illustrative relative differences).
DISTRICTS = {
    "Pauri Garhwal":   {"base_discharge_lpm": 45, "decline_rate": 0.028, "rainfall_mm": 1500, "forest_cover": 0.62},
    "Tehri Garhwal":   {"base_discharge_lpm": 38, "decline_rate": 0.032, "rainfall_mm": 1400, "forest_cover": 0.58},
    "Almora":          {"base_discharge_lpm": 30, "decline_rate": 0.038, "rainfall_mm": 1100, "forest_cover": 0.48},
    "Pithoragarh":     {"base_discharge_lpm": 52, "decline_rate": 0.020, "rainfall_mm": 1600, "forest_cover": 0.66},
    "Nainital":        {"base_discharge_lpm": 40, "decline_rate": 0.030, "rainfall_mm": 1450, "forest_cover": 0.55},
    "Chamoli":         {"base_discharge_lpm": 48, "decline_rate": 0.022, "rainfall_mm": 1550, "forest_cover": 0.64},
    "Bageshwar":       {"base_discharge_lpm": 33, "decline_rate": 0.035, "rainfall_mm": 1200, "forest_cover": 0.50},
    "Rudraprayag":     {"base_discharge_lpm": 42, "decline_rate": 0.026, "rainfall_mm": 1500, "forest_cover": 0.60},
}

SPRINGS_PER_DISTRICT = 6
START_YEAR = 2005
END_YEAR = 2025


def _monthly_rainfall_series(annual_mm: float, n_years: int, rng: np.random.Generator) -> np.ndarray:
    """Monsoon-dominated seasonal rainfall with year-to-year variability."""
    # Relative monthly share of annual rainfall (Jun-Sep monsoon dominant), sums to 1.
    monthly_share = np.array([0.02, 0.02, 0.03, 0.04, 0.06, 0.16, 0.22, 0.20, 0.13, 0.06, 0.03, 0.03])
    months = []
    for _ in range(n_years):
        year_variability = rng.normal(1.0, 0.15)  # some years wetter/drier
        year_total = max(annual_mm * year_variability, annual_mm * 0.5)
        noise = rng.normal(1.0, 0.08, size=12)
        year_months = year_total * monthly_share * noise
        months.extend(year_months.tolist())
    return np.array(months)


def _spring_discharge_series(base_lpm, decline_rate, rainfall_series, rng) -> np.ndarray:
    """
    Discharge (litres/minute) driven by:
    - long-term exponential decline (recharge degradation, land-use change)
    - seasonal lag response to rainfall (springs respond ~1-2 months after rain)
    - random measurement/local noise
    """
    n = len(rainfall_series)
    years_elapsed = np.arange(n) / 12.0
    trend = base_lpm * np.exp(-decline_rate * years_elapsed)

    # Normalize rainfall influence: higher recent rainfall (lagged 1 month) boosts discharge
    rainfall_norm = (rainfall_series - rainfall_series.mean()) / (rainfall_series.std() + 1e-6)
    lagged = np.roll(rainfall_norm, 1)
    lagged[0] = 0
    seasonal_boost = 1 + 0.18 * lagged

    noise = rng.normal(1.0, 0.05, size=n)
    discharge = trend * seasonal_boost * noise
    return np.clip(discharge, 0.5, None)


def generate_dataset(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_years = END_YEAR - START_YEAR + 1
    dates = pd.date_range(f"{START_YEAR}-01-01", periods=n_years * 12, freq="MS")

    rows = []
    for district, props in DISTRICTS.items():
        rainfall = _monthly_rainfall_series(props["rainfall_mm"], n_years, rng)
        for spring_idx in range(1, SPRINGS_PER_DISTRICT + 1):
            spring_id = f"{district.replace(' ', '')}_S{spring_idx}"
            # Slight per-spring variation around district baseline
            base = props["base_discharge_lpm"] * rng.normal(1.0, 0.12)
            decline = props["decline_rate"] * rng.normal(1.0, 0.20)
            discharge = _spring_discharge_series(base, decline, rainfall, rng)

            for d, r, disch in zip(dates, rainfall, discharge):
                rows.append({
                    "district": district,
                    "spring_id": spring_id,
                    "date": d,
                    "rainfall_mm": round(r, 1),
                    "discharge_lpm": round(disch, 2),
                    "forest_cover_index": props["forest_cover"],
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "springs_synthetic.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows across {df['spring_id'].nunique()} springs "
          f"in {df['district'].nunique()} districts.")
    print(df.head())
