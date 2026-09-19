import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" 

AIRPORTS = [
    "EDDF", "EDDM", "EGLL", "EHAM", "LEBL",
    "LEMD", "LFPG", "LIRF", "LTFM", "LSZH",
]

# Scoring is on Jan + Jul 2026, so holding out Jan + Jul 2025 for validation
# mirrors that split as closely as the training data allows.
VALIDATION_MONTHS = [1, 7]


# feature-engineering layer
def add_features(df):

    out = pd.DataFrame(index=df.index)

    sched = df["SCHED_TIME_UTC_mvt"]

    # Decomposed instead of kept as a raw timestamp: a raw datetime64 gets
    # silently cast to an integer nanosecond count by sklearn, which just
    # encodes "how far into 2025 is this" rather than anything about
    # time-of-day congestion or weekly patterns - the things that actually
    # drive taxi-out time.

    out["sched_hour"] = sched.dt.hour
    out["sched_dow"] = sched.dt.dayofweek
    out["sched_month"] = sched.dt.month

    # dep_airport deliberately omitted: these are per-airport files now, so
    # it would be constant (or, after clean_and_split.py's column drop,
    # absent entirely) - zero signal for a per-airport model either way.
    out["aircraft_type"] = df["AIRCRAFT_TYPE_mvt"]
    out["wake_turb"] = df["WK_TBL_CAT_flt"]
    out["operator"] = df["AIRCRAFT_OPERATOR_flt"]
    out["dep_runway"] = df["RUNWAY_mvt"]
    out["has_flt_match"] = df["has_flt_match"]

    # keep the real month (pre-feature-encoding) around for the train/val
    # split, dropped again before fitting
    out["_sched_month_actual"] = sched.dt.month

    return out


def train_one_airport(airport: str, verbose: bool = True):
    data_file = DATA_DIR / f"cleaned_{airport}_2025.feather"
    df = pd.read_feather(data_file)

    model_df = add_features(df)
    target = df.loc[model_df.index, "TAXITIME_SEC_mvt"]

    val_mask = model_df["_sched_month_actual"].isin(VALIDATION_MONTHS)
    model_df = model_df.drop(columns=["_sched_month_actual"])

    train_df = model_df[~val_mask]
    val_df = model_df[val_mask]
    y_train = target[~val_mask]
    y_val = target[val_mask]

    # cast object/category columns explicitly so HGB's native categorical
    # handling picks them up regardless of how they arrived from the feather.
    # AIRCRAFT_OPERATOR_flt runs 300-400+ distinct values per airport (only
    # EGLL stays under 255), over sklearn's native-categorical cap - bucket
    # anything outside the top MAX_CATEGORIES (by train-set frequency, to
    # avoid leaking val-set frequency info) into "OTHER".
    MAX_CATEGORIES = 200
    cat_cols = ["aircraft_type", "wake_turb", "operator", "dep_runway"]
    for col in cat_cols:
        train_str = train_df[col].astype(str)
        val_str = val_df[col].astype(str)
        top = train_str.value_counts().nlargest(MAX_CATEGORIES).index
        # "OTHER" is always a valid category, even if unused in training -
        # covers both bucketed-rare values AND any value seen only in the
        # held-out months (e.g. an aircraft type that first appears in July).
        categories = pd.Index(sorted(set(top) | {"OTHER"}))
        train_df[col] = pd.Categorical(train_str.where(train_str.isin(top), "OTHER"), categories=categories)
        val_df[col] = pd.Categorical(val_str.where(val_str.isin(top), "OTHER"), categories=categories)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        categorical_features="from_dtype",
    )
    model.fit(train_df, y_train)

    predictions = model.predict(val_df)
    rmse = root_mean_squared_error(y_val, predictions)

    if verbose:
        print(f"{airport}: train={len(train_df):>7} val={len(val_df):>6} rmse={rmse:8.2f}s")

    return rmse, model


def main():
    results = {}
    for airport in AIRPORTS:
        rmse, _ = train_one_airport(airport)
        results[airport] = rmse

    overall = np.mean(list(results.values()))
    print(f"\nmean RMSE across airports: {overall:.2f}s")
    return results


if __name__ == "__main__":
    main()
