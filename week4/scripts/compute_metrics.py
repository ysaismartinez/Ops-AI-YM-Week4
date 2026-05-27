"""
Simple monitoring metrics runner.

This script loads baseline and new data, runs the MetricComputer metrics,
prints the results, and writes a JSON report.

Expected repo location:
    week4/scripts/compute_metrics.py

Example usage:
    python3 week4/scripts/compute_metrics.py \
        --baseline data/baseline.csv \
        --new data/feb.csv \
        --output monitoring_metrics.json

If prediction and actual columns exist, the script will also compute accuracy
and accuracy by segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from metric_template import MetricComputer
except ImportError as exc:
    raise ImportError(
        "Could not import MetricComputer. Make sure metric_template.py is in the same folder "
        "as compute_metrics.py, usually week4/scripts/."
    ) from exc


PREDICTION_COLUMN_CANDIDATES = [
    "prediction",
    "predictions",
    "predicted",
    "predicted_pickups",
    "y_pred",
]

ACTUAL_COLUMN_CANDIDATES = [
    "actual",
    "actuals",
    "actual_pickups",
    "y_true",
    "target",
    "label",
]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and pandas objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super().default(obj)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read CSV, Parquet, or JSON data into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, Parquet, JSON, or JSONL.")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column in df matching one of the candidate names."""
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def get_optional_prediction_actual_arrays(new_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return prediction and actual arrays when matching columns are available."""
    prediction_column = find_column(new_df, PREDICTION_COLUMN_CANDIDATES)
    actual_column = find_column(new_df, ACTUAL_COLUMN_CANDIDATES)

    if prediction_column is None or actual_column is None:
        return None, None

    return new_df[prediction_column].to_numpy(), new_df[actual_column].to_numpy()


def evaluate_alerts(metrics: dict[str, Any]) -> list[str]:
    """
    Convert metric values into simple human-readable alerts.

    These thresholds are intentionally simple for the assignment. They can be
    tuned later based on BASELINE_METRICS.md or real production behavior.
    """
    alerts: list[str] = []

    accuracy = metrics.get("accuracy")
    if accuracy is not None and accuracy < 0.80:
        alerts.append(f"Overall accuracy is below threshold: {accuracy:.2%}")

    for zone, value in metrics.get("accuracy_by_zone", {}).items():
        if value < 0.80:
            alerts.append(f"Zone {zone} accuracy is below threshold: {value:.2%}")

    for field, value in metrics.get("null_rates", {}).items():
        if value > 0.01:
            alerts.append(f"Null rate for {field} is above 1%: {value:.2%}")

    ks_result = metrics.get("ks_test", {})
    if ks_result.get("drift_detected"):
        alerts.append(
            f"KS test detected distribution drift in {ks_result.get('column')} "
            f"with p-value {ks_result.get('p_value'):.4g}"
        )

    psi = metrics.get("psi")
    if psi is not None and psi > 0.20:
        alerts.append(f"PSI is above 0.20: {psi:.4f}")

    prediction_distribution = metrics.get("prediction_distribution", {})
    if prediction_distribution.get("collapsed"):
        alerts.append("Prediction distribution appears collapsed or nearly constant")

    freshness = metrics.get("data_freshness", {})
    if freshness.get("is_stale"):
        alerts.append(
            f"Data appears stale. Newest record age is {freshness.get('age_hours'):.2f} hours"
        )

    duplicate_rate = metrics.get("duplicate_rate", {}).get("duplicate_rate")
    if duplicate_rate is not None and duplicate_rate > 0.01:
        alerts.append(f"Duplicate rate is above 1%: {duplicate_rate:.2%}")

    outlier_rate = metrics.get("outlier_rate", {}).get("outlier_rate")
    if outlier_rate is not None and outlier_rate > 0.05:
        alerts.append(f"Outlier rate is above 5%: {outlier_rate:.2%}")

    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute operational monitoring metrics.")
    parser.add_argument("--baseline", required=True, help="Path to baseline data file")
    parser.add_argument("--new", required=True, help="Path to new production data file")
    parser.add_argument("--output", default="monitoring_metrics.json", help="Path to output JSON report")
    args = parser.parse_args()

    print("=" * 70)
    print("MONITORING METRICS")
    print("=" * 70)

    baseline_df = read_dataset(args.baseline)
    new_df = read_dataset(args.new)

    predictions, actuals = get_optional_prediction_actual_arrays(new_df)

    metric_computer = MetricComputer(baseline_df)
    metrics = metric_computer.compute_all_metrics(new_df, predictions=predictions, actuals=actuals)
    alerts = evaluate_alerts(metrics)

    report = {
        "baseline_file": str(args.baseline),
        "new_file": str(args.new),
        "baseline_rows": len(baseline_df),
        "new_rows": len(new_df),
        "metrics": metrics,
        "alerts": alerts,
        "status": "alert" if alerts else "healthy",
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, cls=NumpyEncoder))

    print(json.dumps(report, indent=2, cls=NumpyEncoder))
    print(f"\nSaved monitoring report to: {output_path}")

    if alerts:
        print("\nALERTS DETECTED")
        for alert in alerts:
            print(f"- {alert}")
        return 1

    print("\nNo monitoring alerts detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
