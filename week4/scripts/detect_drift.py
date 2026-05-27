"""
Drift detection script.

Compares Jan 1-15 baseline data against Feb 2-28 new data and reports
4+ distinct drift patterns using KS test, PSI, chi-square tests, and
segment-level performance comparisons.

Expected usage:
    python3 week4/scripts/detect_drift.py

Optional usage with explicit files:
    python3 week4/scripts/detect_drift.py --baseline data/baseline.parquet --new data/new.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


DRIFT_P_VALUE_THRESHOLD = 0.05
PSI_ALERT_THRESHOLD = 0.20
ACCURACY_DROP_THRESHOLD = 0.10
MIN_SEGMENT_ROWS = 20


NUMERIC_FEATURE_CANDIDATES = [
    "trip_count",
    "pickup_count",
    "actual_pickups",
    "predicted_pickups",
    "prediction",
    "fare_amount",
    "trip_distance",
    "passenger_count",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "zone_id",
    "hour",
    "day_of_week",
    "pickup_zone",
    "borough",
]

SEGMENT_CANDIDATES = ["zone_id", "hour", "day_of_week", "pickup_zone", "borough"]


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric and remove missing or infinite values."""
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column from a list of candidate names."""
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _accuracy_from_columns(df: pd.DataFrame) -> pd.Series | None:
    """
    Infer row-level correctness from common prediction and actual column names.

    For classification-like columns, accuracy is exact match.
    For regression-like pickup count columns, accuracy is approximated as whether
    the absolute prediction error is within 20 percent of the actual value, with
    a floor of 1 pickup to avoid division by zero issues.
    """
    prediction_col = _find_column(df, ["prediction", "predictions", "predicted", "predicted_pickups", "y_pred"])
    actual_col = _find_column(df, ["actual", "actuals", "actual_pickups", "y_true", "target", "label"])

    if prediction_col is None or actual_col is None:
        return None

    pred = df[prediction_col]
    actual = df[actual_col]

    pred_numeric = pd.to_numeric(pred, errors="coerce")
    actual_numeric = pd.to_numeric(actual, errors="coerce")

    numeric_rows = pred_numeric.notna() & actual_numeric.notna()
    if numeric_rows.mean() > 0.8:
        denominator = actual_numeric.abs().clip(lower=1)
        relative_error = (pred_numeric - actual_numeric).abs() / denominator
        return (relative_error <= 0.20).astype(float)

    return (pred.astype(str) == actual.astype(str)).astype(float)


def calculate_psi(baseline_values: pd.Series, new_values: pd.Series, bins: int = 10) -> float:
    """Calculate Population Stability Index for two numeric distributions."""
    baseline = _clean_numeric(baseline_values)
    new = _clean_numeric(new_values)

    if baseline.empty or new.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(baseline, quantiles))

    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    new_counts, _ = np.histogram(new, bins=breakpoints)

    baseline_pct = baseline_counts / max(len(baseline), 1)
    new_pct = new_counts / max(len(new), 1)

    epsilon = 1e-6
    baseline_pct = np.clip(baseline_pct, epsilon, None)
    new_pct = np.clip(new_pct, epsilon, None)

    psi = np.sum((new_pct - baseline_pct) * np.log(new_pct / baseline_pct))
    return float(psi)


def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict[str, Any]:
    """
    Detect drift in a single numeric feature using KS test and PSI.

    Returns test results and an interpretation that can be used in the report.
    """
    if feature not in baseline_df.columns or feature not in new_df.columns:
        return {
            "feature": feature,
            "status": "skipped",
            "reason": "feature missing from one of the datasets",
        }

    baseline_values = _clean_numeric(baseline_df[feature])
    new_values = _clean_numeric(new_df[feature])

    if len(baseline_values) < 2 or len(new_values) < 2:
        return {
            "feature": feature,
            "status": "skipped",
            "reason": "not enough numeric observations",
        }

    ks_statistic, p_value = ks_2samp(baseline_values, new_values)
    psi = calculate_psi(baseline_values, new_values)

    drift_detected = bool((p_value < DRIFT_P_VALUE_THRESHOLD) or (psi >= PSI_ALERT_THRESHOLD))

    return {
        "feature": feature,
        "type": "data drift",
        "test": "KS test + PSI",
        "baseline_mean": float(baseline_values.mean()),
        "new_mean": float(new_values.mean()),
        "mean_change_pct": float(((new_values.mean() - baseline_values.mean()) / max(abs(baseline_values.mean()), 1e-6)) * 100),
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
        "psi": float(psi),
        "drift_detected": drift_detected,
        "interpretation": "drift detected" if drift_detected else "no material drift detected",
    }


def detect_categorical_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str) -> dict[str, Any]:
    """Detect categorical distribution drift using chi-square test."""
    if feature not in baseline_df.columns or feature not in new_df.columns:
        return {
            "feature": feature,
            "status": "skipped",
            "reason": "feature missing from one of the datasets",
        }

    baseline_counts = baseline_df[feature].fillna("__MISSING__").astype(str).value_counts()
    new_counts = new_df[feature].fillna("__MISSING__").astype(str).value_counts()
    categories = sorted(set(baseline_counts.index).union(set(new_counts.index)))

    if len(categories) < 2:
        return {
            "feature": feature,
            "status": "skipped",
            "reason": "not enough categories",
        }

    contingency = np.array([
        [baseline_counts.get(category, 0) for category in categories],
        [new_counts.get(category, 0) for category in categories],
    ])

    chi2, p_value, _, _ = chi2_contingency(contingency)
    drift_detected = bool(p_value < DRIFT_P_VALUE_THRESHOLD)

    baseline_share = baseline_counts / max(baseline_counts.sum(), 1)
    new_share = new_counts / max(new_counts.sum(), 1)
    share_changes = []
    for category in categories:
        old_share = float(baseline_share.get(category, 0))
        current_share = float(new_share.get(category, 0))
        share_changes.append({
            "category": category,
            "baseline_share": old_share,
            "new_share": current_share,
            "absolute_change": abs(current_share - old_share),
        })
    share_changes = sorted(share_changes, key=lambda item: item["absolute_change"], reverse=True)[:5]

    return {
        "feature": feature,
        "type": "data drift",
        "test": "chi-square",
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "largest_share_changes": share_changes,
        "interpretation": "categorical drift detected" if drift_detected else "no material categorical drift detected",
    }


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect concept drift by comparing inferred accuracy by segment.

    The function looks for prediction and actual columns. If they are available,
    it calculates row-level correctness and compares segment accuracy between
    baseline and new data. A segment is flagged when accuracy drops by at least
    ACCURACY_DROP_THRESHOLD.
    """
    baseline_accuracy = _accuracy_from_columns(baseline_df)
    new_accuracy = _accuracy_from_columns(new_df)

    if baseline_accuracy is None or new_accuracy is None:
        return {
            "type": "concept drift",
            "status": "skipped",
            "reason": "prediction and actual columns were not found",
            "findings": [],
        }

    baseline_work = baseline_df.copy()
    new_work = new_df.copy()
    baseline_work["__accuracy__"] = baseline_accuracy
    new_work["__accuracy__"] = new_accuracy

    findings = []
    for segment in SEGMENT_CANDIDATES:
        if segment not in baseline_work.columns or segment not in new_work.columns:
            continue

        baseline_grouped = baseline_work.groupby(segment)["__accuracy__"].agg(["mean", "count"])
        new_grouped = new_work.groupby(segment)["__accuracy__"].agg(["mean", "count"])
        common_segments = set(baseline_grouped.index).intersection(set(new_grouped.index))

        for value in common_segments:
            baseline_count = int(baseline_grouped.loc[value, "count"])
            new_count = int(new_grouped.loc[value, "count"])
            if baseline_count < MIN_SEGMENT_ROWS or new_count < MIN_SEGMENT_ROWS:
                continue

            baseline_acc = float(baseline_grouped.loc[value, "mean"])
            new_acc = float(new_grouped.loc[value, "mean"])
            drop = baseline_acc - new_acc

            if drop >= ACCURACY_DROP_THRESHOLD:
                findings.append({
                    "segment": segment,
                    "segment_value": str(value),
                    "type": "concept drift",
                    "baseline_accuracy": baseline_acc,
                    "new_accuracy": new_acc,
                    "accuracy_drop": float(drop),
                    "baseline_rows": baseline_count,
                    "new_rows": new_count,
                    "impact": f"accuracy dropped by {drop:.1%}",
                })

    findings = sorted(findings, key=lambda item: item["accuracy_drop"], reverse=True)

    return {
        "type": "concept drift",
        "test": "segment accuracy comparison",
        "threshold": ACCURACY_DROP_THRESHOLD,
        "drift_detected": len(findings) > 0,
        "findings": findings,
    }


def detect_data_quality_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect null rate, duplicate rate, and outlier rate changes."""
    findings: list[dict[str, Any]] = []

    common_columns = [col for col in baseline_df.columns if col in new_df.columns]
    for column in common_columns:
        baseline_null_rate = float(baseline_df[column].isna().mean())
        new_null_rate = float(new_df[column].isna().mean())
        increase = new_null_rate - baseline_null_rate
        if increase >= 0.05 and new_null_rate >= 0.05:
            findings.append({
                "pattern": "null rate increase",
                "feature": column,
                "type": "data quality issue",
                "baseline_value": baseline_null_rate,
                "new_value": new_null_rate,
                "magnitude": increase,
                "impact": "missing values can reduce feature reliability and prediction quality",
            })

    baseline_duplicate_rate = float(baseline_df.duplicated().mean())
    new_duplicate_rate = float(new_df.duplicated().mean())
    duplicate_increase = new_duplicate_rate - baseline_duplicate_rate
    if duplicate_increase >= 0.02 and new_duplicate_rate >= 0.02:
        findings.append({
            "pattern": "duplicate rate increase",
            "type": "data quality issue",
            "baseline_value": baseline_duplicate_rate,
            "new_value": new_duplicate_rate,
            "magnitude": duplicate_increase,
            "impact": "duplicates can overrepresent repeated trips and bias monitoring metrics",
        })

    for column in common_columns:
        baseline_values = _clean_numeric(baseline_df[column])
        new_values = _clean_numeric(new_df[column])
        if len(baseline_values) < 10 or len(new_values) < 10:
            continue
        q1 = baseline_values.quantile(0.25)
        q3 = baseline_values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        baseline_outlier_rate = float(((baseline_values < lower) | (baseline_values > upper)).mean())
        new_outlier_rate = float(((new_values < lower) | (new_values > upper)).mean())
        outlier_increase = new_outlier_rate - baseline_outlier_rate
        if outlier_increase >= 0.05 and new_outlier_rate >= 0.05:
            findings.append({
                "pattern": "outlier rate increase",
                "feature": column,
                "type": "data quality issue",
                "baseline_value": baseline_outlier_rate,
                "new_value": new_outlier_rate,
                "magnitude": outlier_increase,
                "impact": "outliers may indicate abnormal demand spikes, bad records, or upstream data errors",
            })

    return findings


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV, Parquet, or JSON dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_default_file(keywords: list[str]) -> Path | None:
    """Search common project folders for a likely baseline or new data file."""
    search_roots = [Path("."), Path("data"), Path("week4/data"), Path("week4")]
    extensions = ["*.csv", "*.parquet", "*.pq", "*.json", "*.jsonl"]

    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for extension in extensions:
            candidates.extend(root.rglob(extension))

    for path in candidates:
        name = path.name.lower()
        if any(keyword.lower() in name for keyword in keywords):
            return path
    return None


def summarize_findings(feature_results: list[dict[str, Any]], categorical_results: list[dict[str, Any]], concept_results: dict[str, Any], quality_findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a report-ready list of drift patterns."""
    patterns: list[dict[str, Any]] = []

    for result in feature_results:
        if result.get("drift_detected"):
            patterns.append({
                "what_drifted": result["feature"],
                "type": result["type"],
                "statistical_evidence": f"KS statistic={result['ks_statistic']:.4f}, p-value={result['p_value']:.4g}, PSI={result['psi']:.4f}",
                "magnitude": f"mean changed {result['mean_change_pct']:.1f}%",
                "impact": "input distribution changed, which can reduce prediction reliability",
            })

    for result in categorical_results:
        if result.get("drift_detected"):
            largest_change = result.get("largest_share_changes", [{}])[0]
            patterns.append({
                "what_drifted": result["feature"],
                "type": result["type"],
                "statistical_evidence": f"chi-square={result['chi_square']:.4f}, p-value={result['p_value']:.4g}",
                "magnitude": f"largest category share change={largest_change.get('absolute_change', 0):.1%}",
                "impact": "segment mix changed, so global model metrics may hide local degradation",
            })

    for finding in concept_results.get("findings", [])[:10]:
        patterns.append({
            "what_drifted": f"{finding['segment']}={finding['segment_value']}",
            "type": "concept drift",
            "statistical_evidence": "segment accuracy comparison",
            "magnitude": f"accuracy dropped from {finding['baseline_accuracy']:.1%} to {finding['new_accuracy']:.1%}",
            "impact": finding["impact"],
        })

    for finding in quality_findings[:10]:
        patterns.append({
            "what_drifted": finding.get("feature", finding["pattern"]),
            "type": finding["type"],
            "statistical_evidence": finding["pattern"],
            "magnitude": f"baseline={finding['baseline_value']:.2%}, new={finding['new_value']:.2%}",
            "impact": finding["impact"],
        })

    return patterns


def main() -> int:
    """Main drift detection analysis."""
    parser = argparse.ArgumentParser(description="Detect drift between baseline and new data.")
    parser.add_argument("--baseline", type=str, default=None, help="Path to Jan 1-15 baseline data")
    parser.add_argument("--new", type=str, default=None, help="Path to Feb 2-28 new data")
    parser.add_argument("--output", type=str, default="drift_report.json", help="Path for JSON report")
    args = parser.parse_args()

    print("=" * 70)
    print("DRIFT DETECTION")
    print("=" * 70)

    baseline_path = Path(args.baseline) if args.baseline else find_default_file(["baseline", "jan", "jan_1_15", "jan1_15"])
    new_path = Path(args.new) if args.new else find_default_file(["new", "feb", "feb_2_28", "feb2_28", "current"])

    if baseline_path is None or new_path is None:
        print("Could not automatically find baseline and new data files.")
        print("Run with explicit paths, for example:")
        print("python3 week4/scripts/detect_drift.py --baseline data/baseline.csv --new data/feb.csv")
        return 1

    print(f"Baseline data: {baseline_path}")
    print(f"New data:      {new_path}")

    baseline_df = load_dataframe(baseline_path)
    new_df = load_dataframe(new_path)

    print(f"Baseline rows: {len(baseline_df):,}")
    print(f"New rows:      {len(new_df):,}")
    print()

    numeric_features = [feature for feature in NUMERIC_FEATURE_CANDIDATES if feature in baseline_df.columns and feature in new_df.columns]
    if not numeric_features:
        numeric_features = [column for column in baseline_df.select_dtypes(include=[np.number]).columns if column in new_df.columns]

    categorical_features = [feature for feature in CATEGORICAL_FEATURE_CANDIDATES if feature in baseline_df.columns and feature in new_df.columns]

    feature_results = [detect_feature_drift(baseline_df, new_df, feature) for feature in numeric_features]
    categorical_results = [detect_categorical_drift(baseline_df, new_df, feature) for feature in categorical_features]
    concept_results = detect_concept_drift_by_segment(baseline_df, new_df)
    quality_findings = detect_data_quality_drift(baseline_df, new_df)

    patterns = summarize_findings(feature_results, categorical_results, concept_results, quality_findings)

    print("Detected drift patterns")
    print("-" * 70)
    if not patterns:
        print("No major drift patterns exceeded the configured thresholds.")
    else:
        for index, pattern in enumerate(patterns[:12], start=1):
            print(f"{index}. What drifted: {pattern['what_drifted']}")
            print(f"   Type: {pattern['type']}")
            print(f"   Evidence: {pattern['statistical_evidence']}")
            print(f"   Magnitude: {pattern['magnitude']}")
            print(f"   Impact: {pattern['impact']}")
            print()

    report = {
        "baseline_path": str(baseline_path),
        "new_path": str(new_path),
        "baseline_rows": len(baseline_df),
        "new_rows": len(new_df),
        "numeric_feature_results": feature_results,
        "categorical_feature_results": categorical_results,
        "concept_drift_results": concept_results,
        "data_quality_findings": quality_findings,
        "summary_patterns": patterns,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {output_path}")

    if len(patterns) >= 4:
        print(f"PASS: Found {len(patterns)} drift patterns. Requirement was 4+.")
    else:
        print(f"WARNING: Found {len(patterns)} drift patterns. Requirement was 4+.")
        print("You may need to lower thresholds or inspect additional features if the dataset is small.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
