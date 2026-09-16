"""
Data loading and preprocessing pipeline for green hydrogen production.
Loads an NLR wind electrolysis dataset, maps it to the 10-channel
input format described in the methodology, computes LCOH targets, and creates
temporal sliding windows for Transformer training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ── Column mapping: NLR dataset → 10-channel methodology format ─────────────────
NLR_COLUMNS = {
    "wind_speed":        "Wind_Speed_ms",       # derived from GE turbine power
    "turbulence":        "Turbulence_pct",       # estimated
    "dc_stack_current":  "H2E_f_NLR_CurrentCmd (A)",
    "current_ramp_rate": "Current_Ramp_As",     # from experiment metadata
    "stack_temperature": "H2E_f_TE218_Temp",    # representative stack temp sensor
    "stack_pressure":    "H2E_f_PT264_Pressure", # stack inlet pressure
    "power_consumption": "H2E_n_PSU_A_Power (kW)",
    "efficiency_kwh_kg": "Efficiency (kWh/kg)",
    "h2_history":       "h2_history",
    "experiment_progress": "experiment_progress",
}
TARGET_COLUMNS = {
    "h2_production": "IVAL_f_FM011_Flow",   # H2 mass flow rate (ground truth)
}
LCOH_COLUMNS = {
    "efficiency_kwh_kg": "Efficiency (kWh/kg)",
}

INPUT_KEYS  = list(NLR_COLUMNS.keys())
TARGET_KEY  = "h2_production"
LCOH_KEY    = "lcost"
OUTPUT_KEYS = [TARGET_KEY, LCOH_KEY]

# EU Hydrogen Observatory LCOH assumptions (methodology §II.B, ref [2])
LCOH_PARAMS = dict(
    capex_usd_kg     = 700.0,   # $/kg H2 – PEM stack CAPEX
    project_life_yr  = 10,
    water_cost_usd_m3= 0.004,   # $/L → $/m3
    electricity_price= 0.06,    # $/kWh – generic renewable assumption
    capacity_factor  = 0.45,
    stack_degradation= 0.03,    # 3 %/yr efficiency loss
)


def derive_wind_speed(power_kw: pd.Series) -> pd.Series:
    """
    Approximate hub-height wind speed from turbine power using a simplified
    power curve for a GE 1.5 MW turbine.
    Cut-in ~3.5 m/s, rated ~11 m/s, rated power 1500 kW.
    """
    rated_power = 1500.0   # kW
    rated_ws    = 11.0     # m/s
    cutin_ws    = 3.5      # m/s
    # Inverse power curve: p = min(1, max(0, (v-cutin)/(rated-cutin)))
    # => v = p*(rated-cutin) + cutin
    p = power_kw.clip(0, rated_power) / rated_power
    ws = p * (rated_ws - cutin_ws) + cutin_ws
    return ws.clip(cutin_ws, rated_ws + 1)


def compute_lcoh(row: pd.Series, params: dict) -> float:
    """
    Compute Levelized Cost of Hydrogen ($/kg) using a simplified LCOH model
    based on the EU Hydrogen Observatory methodology.
    LCOH ≈ (CAPEX/N + OPEX + water + electricity) / H2_annual_kg
    where N is project lifetime in years.
    """
    capex     = params["capex_usd_kg"]
    n_yr      = params["project_life_yr"]
    water     = params["water_cost_usd_m3"]
    elec_price= params["electricity_price"]
    cf        = params["capacity_factor"]
    degr      = params["stack_degradation"]

    eff_kwh_kg = max(row["efficiency_kwh_kg"], 1.0)   # kWh/kg H2
    h2_rate    = max(row[TARGET_KEY], 0.001)          # kg/hr

    # Annual H2 production (hours × capacity_factor)
    annual_hrs = 8760 * cf
    annual_kg  = h2_rate * annual_hrs

    # Electricity cost per kg H2
    elec_cost_kg = eff_kwh_kg * elec_price

    # Water cost per kg (~9 L water / 1 kg H2)
    water_cost_kg = 9.0 * water

    # Annualised CAPEX per kg
    capex_ann_kg  = capex / n_yr

    # Degradation penalty (simple additive)
    degr_cost_kg = capex * degr * n_yr / 2 / annual_kg  # mid-life degradation

    lc = capex_ann_kg + elec_cost_kg + water_cost_kg + degr_cost_kg
    return max(lc, 0.5)   # floor at $0.50/kg


def load_and_preprocess(raw_csv: str, lcoh_params: dict = LCOH_PARAMS) -> pd.DataFrame:
    """
    Load raw NLR CSV and preprocess.
    Uses the raw 1 Hz telemetry directly (no resampling) for maximum data fidelity.
    Maps NLR columns to the 10-channel methodology input format, computes derived
    features (wind speed, turbulence), estimates LCOH, and handles missing values.

    Args:
        raw_csv: Path to combined_historical_wind_experiments.csv
        lcoh_params: LCOH computation parameters
    """
    print(f"[data] Loading raw CSV: {raw_csv}")
    # NLR releases use slightly different headers between submissions. Read the
    # first column as the timestamp and resolve the remaining columns by aliases.
    df = pd.read_csv(raw_csv)
    timestamp_col = df.columns[0]
    df = df.rename(columns={timestamp_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def resolve(*names):
        for name in names:
            if name in df.columns:
                return name
        raise KeyError(f"None of these columns were found: {names}")

    current_col = resolve("H2E_f_NLR_CurrentCmd (A)", "H2E_f_NLR_CurrentCmd (%)")
    power_col = resolve("H2E_n_PSU_A_Power (kW)", "H2E_n_PSU_A_Power")
    wind_power_col = resolve(
        "GE 1.5 MW Wind Turbine Power (kW)", "Wind Turbine Power (kW)"
    )
    h2_col = resolve("IVAL_f_FM011_Flow (kg/hr)", "IVAL_f_FM011_Flow")
    temp_col = resolve("H2E_f_TE218_Temp")
    pressure_col = resolve("H2E_f_PT264_Pressure")
    efficiency_col = resolve("Efficiency (kWh/kg)")

    # ── Experiment metadata ─────────────────────────────────────────────────────
    # The experiment name ends in -200.csv or -400.csv across NLR wind releases.
    df["Current_Ramp_As"] = (
        df["Experiment"].astype(str).str.extract(r"-(\d+)(?:\.csv)?$", expand=False)
        .astype(float)
    )

    # ── Derive wind speed from turbine power ──────────────────────────────────
    df["Wind_Speed_ms"] = derive_wind_speed(pd.to_numeric(df[wind_power_col], errors="coerce"))

    # ── Estimate turbulence intensity (TI %) ─────────────────────────────────
    ws = df["Wind_Speed_ms"]
    # Use rolling std as proxy (at 1 Hz, 60-sample window = 60 s)
    ws_roll_std = ws.rolling(60, min_periods=10).std()
    ti_proxy    = (ws_roll_std / (ws + 0.1) * 100).clip(0, 50)
    # Fill start with ffill first (data-start NaNs), then bfill (data-end NaNs)
    df["Turbulence_pct"] = ti_proxy.ffill(limit=60).bfill(limit=60)

    # ── Build feature DataFrame ────────────────────────────────────────────────
    feat_df = pd.DataFrame()
    feat_df["timestamp"]           = df["timestamp"]
    feat_df["wind_speed"]         = df["Wind_Speed_ms"]
    feat_df["turbulence"]         = df["Turbulence_pct"]
    feat_df["dc_stack_current"]   = df[current_col]
    feat_df["current_ramp_rate"]  = df["Current_Ramp_As"]
    feat_df["stack_temperature"]  = df[temp_col]
    feat_df["stack_pressure"]     = df[pressure_col]
    feat_df["power_consumption"]  = df[power_col]
    feat_df["h2_production"]     = df[h2_col]
    feat_df["efficiency_kwh_kg"] = df[efficiency_col]
    feat_df["h2_history"]       = df[h2_col]
    exp_group = df["Experiment"].fillna("unknown")
    feat_df["experiment_name"] = exp_group
    feat_df["experiment_position"] = exp_group.groupby(exp_group).cumcount()
    feat_df["experiment_progress"] = exp_group.groupby(exp_group).cumcount() / exp_group.groupby(exp_group).transform("size").clip(lower=2).sub(1)

    # Use raw 1 Hz data directly (no resampling) for maximum fidelity
    feat_df = feat_df.set_index("timestamp")
    numeric_cols = [c for c in feat_df.columns if c not in ("timestamp", "experiment_name")]
    feat_df[numeric_cols] = feat_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    feat_df[numeric_cols] = feat_df[numeric_cols].ffill(limit=5).interpolate(method="linear")
    feat_df = feat_df.dropna(subset=["h2_production"]).reset_index()
    freq_desc = "1 Hz"

    print(f"[data] After preprocessing: {len(feat_df)} samples @ {freq_desc}")

    # ── Compute LCOH target ───────────────────────────────────────────────────
    lc_values = []
    for _, row in feat_df.iterrows():
        lc_values.append(compute_lcoh(row, lcoh_params))
    feat_df[LCOH_KEY] = lc_values

    # ── Add experiment indicator for split stratification ─────────────────────
    feat_df["exp_flag"] = pd.factorize(feat_df["experiment_name"])[0]
    feat_df["phase_id"] = (feat_df["experiment_progress"] * 10).astype(int).clip(0, 9)
    feat_df["segment_id"] = feat_df["exp_flag"] * 100 + feat_df["phase_id"]

    return feat_df


class HydrogenDataset(Dataset):
    """
    Sliding-window dataset for multivariate time-series.
    Each sample returns:
      - input  : (T, D)  look-back window of D input channels
      - target : (2,)    [h2_production, LCOH] for the forecast horizon
    """

    def __init__(
        self,
        features: pd.DataFrame,
        lookback: int = 1440,   # 24 h at 1-min cadence
        horizon:  int = 60,    # 1 h forecast
        stride:   int = 15,    # 15-min stride for more samples
        input_cols: list = INPUT_KEYS,
        target_cols: list = OUTPUT_KEYS,
        scaler: object = None,
        fit_scaler: bool = False,
    ):
        self.lookback   = lookback
        self.horizon    = horizon
        self.input_cols = input_cols
        self.target_cols= target_cols

        # ── Normalization ─────────────────────────────────────────────────────
        X_raw = features[input_cols].values.astype(np.float32)
        y_raw = features[target_cols].values.astype(np.float32)

        if scaler is None:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            if fit_scaler:
                X_scaled = self.scaler_X.fit_transform(X_raw)
                y_scaled = self.scaler_y.fit_transform(y_raw)
            else:
                raise ValueError("fit_scaler=True requires scaler=None")
        else:
            self.scaler_X, self.scaler_y = scaler
            X_scaled = self.scaler_X.transform(X_raw)
            y_scaled = self.scaler_y.transform(y_raw)

        self.X = X_scaled
        self.y = y_scaled
        experiment_ids = features["segment_id"].to_numpy() if "segment_id" in features else (features["exp_flag"].to_numpy() if "exp_flag" in features else None)

        # ── Build sliding windows without crossing experiment boundaries ───────
        self.windows = []
        self.window_metadata = []
        n = len(self.X)
        for start in range(0, n - lookback - horizon + 1, stride):
            end_in  = start + lookback
            tgt_end = end_in + horizon
            if experiment_ids is not None:
                if not np.all(experiment_ids[start:tgt_end] == experiment_ids[start]):
                    continue
            self.windows.append((start, end_in, tgt_end))
            target_row = features.iloc[tgt_end - 1]
            self.window_metadata.append({
                "experiment": str(target_row.get("experiment_name", "unknown")),
                "experiment_position": int(target_row.get("experiment_position", tgt_end - 1)),
                "target_timestamp": str(target_row.get("timestamp", "")),
            })
        print(f"[dataset] Built {len(self.windows)} windows "
              f"(lookback={lookback}, horizon={horizon}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end_in, tgt_end = self.windows[idx]
        x = torch.from_numpy(self.X[start:end_in])          # (T, D)
        y = torch.from_numpy(self.y[tgt_end - 1])            # (2,) last step of horizon
        return x, y


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
    test_frac:  float = 0.20,
    stratify_col: str = "segment_id",
) -> tuple:
    """Split by operating phases so each set sees the full production range.

    This is a phase-balanced benchmark rather than a strict future-only split.
    Windows never cross segment boundaries because segment_id is retained.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    if "exp_flag" not in df.columns:
        n = len(df)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()

    # Ten equal-duration phases per experiment: 60/20/20 balanced allocation.
    phase_sets = {
        # Interleave phases so every split contains high, medium, and low H2.
        "train": {0, 3, 6, 9},
        "val": {1, 4, 7},
        "test": {2, 5, 8},
    }
    parts = {name: [] for name in phase_sets}
    for _, group in df.groupby("exp_flag", sort=False):
        for name, phases in phase_sets.items():
            part = group[group["phase_id"].isin(phases)]
            if not part.empty:
                parts[name].append(part)

    return tuple(pd.concat(parts[name], ignore_index=True) for name in ("train", "val", "test"))

def build_dataloaders(
    df: pd.DataFrame,
    lookback: int = 1440,
    horizon:  int = 60,
    stride:   int = 15,
    batch_size: int = 32,
    num_workers: int = 0,
    # For small datasets: share scaler by using combined train+val for fit,
    # then re-split windows (train/val share the same scaler)
    use_combined_scaler: bool = False,
) -> tuple:
    """Build train / val / test DataLoaders from a preprocessed DataFrame.

    For small datasets, the scaler is fit on all non-test data (train+val)
    to ensure sufficient samples for normalization.
    """
    train_df, val_df, test_df = temporal_split(df)
    print(f"[split] H2 means: train={train_df[TARGET_KEY].mean():.3f}, "
          f"val={val_df[TARGET_KEY].mean():.3f}, test={test_df[TARGET_KEY].mean():.3f}")

    # Fit scaler on training split only (no data leakage)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(train_df[INPUT_KEYS].values.astype(np.float64))
    scaler_y.fit(train_df[OUTPUT_KEYS].values.astype(np.float64))

    if use_combined_scaler:
        # Re-fit on train+val for small datasets
        combined = pd.concat([train_df, val_df], ignore_index=True)
        scaler_X.fit(combined[INPUT_KEYS].values.astype(np.float64))
        scaler_y.fit(combined[OUTPUT_KEYS].values.astype(np.float64))

    scaler = (scaler_X, scaler_y)

    train_ds = HydrogenDataset(train_df, lookback, horizon, stride,
                               fit_scaler=True, scaler=(scaler_X, scaler_y))
    val_ds   = HydrogenDataset(val_df,   lookback, horizon, stride,
                               scaler=scaler)
    test_ds  = HydrogenDataset(test_df,  lookback, horizon, stride,
                               scaler=scaler)

    def loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=False)

    return loader(train_ds), loader(val_ds), loader(test_ds), scaler


if __name__ == "__main__":
    raw = r"E:\pythonProject\yfel\data\raw\combined_historical_wind_experiments.csv"
    df  = load_and_preprocess(raw)
    print(df.head())
    print(df.describe())
