"""
Monitoring metrics implementation.

This module computes monitoring metrics for an operational ML workflow.
The metrics cover model performance, data quality, data drift, prediction health,
and infrastructure freshness.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    def __init__(self, baseline_df: pd.DataFrame):
        """Initialize with baseline data."""
        self.baseline_df = baseline_df.copy()

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        """Convert input values to a flattened numpy array."""
        if values is None:
            return np.array([])
        return np.asarray(values).reshape(-1)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert numpy values to normal Python floats for JSON serialization."""
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Return the first candidate column that exists in the dataframe."""
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _numeric_monitoring_column(self, new_df: pd.DataFrame) -> str:
        """
        Pick the main numeric column for drift checks.

        The assignment example mentions trip_count. If that column is present,
        it is used. Otherwise, the method falls back to common target columns,
        then the first numeric column shared by baseline and new data.
        """
        preferred_columns = [
            "trip_count",
            "actual_pickups",
            "pickups",
            "pickup_count",
            "fare_amount",
            "prediction",
            "predicted_pickups",
        ]

        for column in preferred_columns:
            if column in self.baseline_df.columns and column in new_df.columns:
                return column

        baseline_numeric = set(self.baseline_df.select_dtypes(include=[np.number]).columns)
        new_numeric = list(new_df.select_dtypes(include=[np.number]).columns)
        shared_numeric = [column for column in new_numeric if column in baseline_numeric]

        if not shared_numeric:
            raise ValueError("No shared numeric column found for drift checks.")

        return shared_numeric[0]

    def metric_1_accuracy(
        self,
        new_df: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """
        Metric 1: Overall Accuracy

        Computation: fraction of records where prediction equals actual.
        Baseline: expected healthy value from BASELINE_METRICS.md.
        Alert threshold: alert if below the baseline threshold, commonly below 80 percent.
        Frequency: daily after ground truth is available.
        Segmentation: global.
        """
        predictions = self._to_numpy(predictions)
        actuals = self._to_numpy(actuals)

        if predictions.size == 0 or actuals.size == 0:
            return np.nan

        if predictions.size != actuals.size:
            raise ValueError("Predictions and actuals must have the same length.")

        return float(np.mean(predictions == actuals))

    def metric_2_accuracy_by_zone(
        self,
        new_df: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> dict:
        """
        Metric 2: Accuracy by Zone

        Computation: for each zone, fraction of records where prediction equals actual.
        Baseline: expected healthy zone accuracy from BASELINE_METRICS.md.
        Alert threshold: alert if any zone falls below 80 percent.
        Frequency: daily after ground truth is available.
        Segmentation: per zone plus global rollup.
        """
        predictions = self._to_numpy(predictions)
        actuals = self._to_numpy(actuals)

        if predictions.size == 0 or actuals.size == 0:
            return {}

        if predictions.size != actuals.size:
            raise ValueError("Predictions and actuals must have the same length.")

        if len(new_df) != predictions.size:
            raise ValueError("new_df, predictions, and actuals must have the same number of rows.")

        zone_column = self._find_first_existing_column(
            new_df,
            ["zone_id", "location_id", "pickup_zone", "PULocationID", "zone"],
        )

        if zone_column is None:
            return {"global": float(np.mean(predictions == actuals))}

        metric_df = new_df[[zone_column]].copy()
        metric_df["correct"] = predictions == actuals

        by_zone = metric_df.groupby(zone_column)["correct"].mean()
        return {str(zone): float(accuracy) for zone, accuracy in by_zone.items()}

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 3: Null Rates for Critical Fields

        Computation: null count divided by row count for critical fields.
        Baseline: near 0 percent for required operational fields.
        Alert threshold: alert if any critical field exceeds 1 percent nulls.
        Frequency: every monitoring run.
        Segmentation: global, with optional per field inspection.
        """
        preferred_columns = [
            "zone_id",
            "location_id",
            "pickup_zone",
            "PULocationID",
            "trip_count",
            "actual_pickups",
            "predicted_pickups",
            "pickup_datetime",
            "timestamp",
            "date",
            "hour",
        ]

        critical_columns = [column for column in preferred_columns if column in new_df.columns]

        if not critical_columns:
            critical_columns = list(new_df.columns)

        return {
            column: float(new_df[column].isna().mean())
            for column in critical_columns
        }

    def metric_4_ks_test(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 4: KS Test for Distribution Shift

        Computation: two sample Kolmogorov Smirnov test comparing baseline and new data.
        Baseline: high p value and low statistic when distributions are stable.
        Alert threshold: alert when p value is below 0.05.
        Frequency: every monitoring run.
        Segmentation: global, using the primary numeric monitoring column.
        """
        column = self._numeric_monitoring_column(new_df)

        baseline_values = pd.to_numeric(self.baseline_df[column], errors="coerce").dropna()
        new_values = pd.to_numeric(new_df[column], errors="coerce").dropna()

        if baseline_values.empty or new_values.empty:
            return {
                "column": column,
                "statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": False,
            }

        result = ks_2samp(baseline_values, new_values)

        return {
            "column": column,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "drift_detected": bool(result.pvalue < 0.05),
        }

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10) -> float:
        """
        Metric 5: Population Stability Index

        Computation: compare baseline and new distributions using fixed baseline bins.
        Baseline: PSI below 0.10 is generally stable.
        Alert threshold: alert at PSI above 0.20.
        Frequency: every monitoring run.
        Segmentation: global.
        """
        column = self._numeric_monitoring_column(new_df)

        baseline_values = pd.to_numeric(self.baseline_df[column], errors="coerce").dropna().to_numpy()
        new_values = pd.to_numeric(new_df[column], errors="coerce").dropna().to_numpy()

        if baseline_values.size == 0 or new_values.size == 0:
            return np.nan

        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(np.quantile(baseline_values, quantiles))

        if bin_edges.size < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        baseline_counts, _ = np.histogram(baseline_values, bins=bin_edges)
        new_counts, _ = np.histogram(new_values, bins=bin_edges)

        epsilon = 1e-6
        baseline_percents = baseline_counts / max(baseline_counts.sum(), 1)
        new_percents = new_counts / max(new_counts.sum(), 1)

        baseline_percents = np.clip(baseline_percents, epsilon, None)
        new_percents = np.clip(new_percents, epsilon, None)

        psi = np.sum((new_percents - baseline_percents) * np.log(new_percents / baseline_percents))
        return float(psi)

    def metric_6_prediction_distribution(self, predictions: np.ndarray) -> dict:
        """
        Metric 6: Prediction Distribution Shift

        Computation: calculate prediction mean, standard deviation, min, max, and collapse flag.
        Baseline: prediction distribution should resemble training or historical production behavior.
        Alert threshold: alert if prediction standard deviation is near zero or all predictions are identical.
        Frequency: every monitoring run.
        Segmentation: global.
        """
        predictions = self._to_numpy(predictions)

        if predictions.size == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "unique_values": 0,
                "collapsed": False,
            }

        unique_values = len(np.unique(predictions))
        std = float(np.std(predictions))

        return {
            "mean": float(np.mean(predictions)),
            "std": std,
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "unique_values": int(unique_values),
            "collapsed": bool(std < 1e-6 or unique_values <= 1),
        }

    def metric_7_data_freshness(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 7: Data Freshness

        Computation: age of the newest timestamp in minutes and hours.
        Baseline: newest record should arrive within the expected ingestion window.
        Alert threshold: alert if data is older than 120 minutes for hourly monitoring.
        Frequency: every monitoring run.
        Segmentation: global.
        """
        timestamp_column = self._find_first_existing_column(
            new_df,
            [
                "timestamp",
                "event_time",
                "pickup_datetime",
                "datetime",
                "created_at",
                "date",
            ],
        )

        if timestamp_column is None:
            return {
                "timestamp_column": None,
                "latest_timestamp": None,
                "age_minutes": np.nan,
                "age_hours": np.nan,
                "is_stale": False,
            }

        timestamps = pd.to_datetime(new_df[timestamp_column], errors="coerce", utc=True).dropna()

        if timestamps.empty:
            return {
                "timestamp_column": timestamp_column,
                "latest_timestamp": None,
                "age_minutes": np.nan,
                "age_hours": np.nan,
                "is_stale": False,
            }

        latest_timestamp = timestamps.max()
        now = pd.Timestamp.now(tz=timezone.utc)
        age_minutes = (now - latest_timestamp).total_seconds() / 60

        return {
            "timestamp_column": timestamp_column,
            "latest_timestamp": latest_timestamp.isoformat(),
            "age_minutes": float(age_minutes),
            "age_hours": float(age_minutes / 60),
            "is_stale": bool(age_minutes > 120),
        }

    def metric_8_duplicate_rate(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 8: Duplicate Rate

        Computation: exact duplicate row count divided by total row count.
        Baseline: near 0 percent duplicate rows.
        Alert threshold: alert if duplicate rate exceeds 1 percent.
        Frequency: every monitoring run.
        Segmentation: global.
        """
        row_count = len(new_df)

        if row_count == 0:
            return {
                "duplicate_count": 0,
                "duplicate_rate": 0.0,
                "row_count": 0,
            }

        duplicate_count = int(new_df.duplicated().sum())
        duplicate_rate = duplicate_count / row_count

        return {
            "duplicate_count": duplicate_count,
            "duplicate_rate": float(duplicate_rate),
            "row_count": int(row_count),
        }

    def metric_9_outlier_rate(self, new_df: pd.DataFrame) -> dict:
        """
        Metric 9: Outlier Rate

        Computation: percentage of records outside baseline IQR bounds.
        Baseline: low outlier rate under historical distribution.
        Alert threshold: alert if outlier rate exceeds 5 percent.
        Frequency: every monitoring run.
        Segmentation: global.
        """
        column = self._numeric_monitoring_column(new_df)

        baseline_values = pd.to_numeric(self.baseline_df[column], errors="coerce").dropna()
        new_values = pd.to_numeric(new_df[column], errors="coerce").dropna()

        if baseline_values.empty or new_values.empty:
            return {
                "column": column,
                "outlier_count": 0,
                "outlier_rate": np.nan,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
            }

        q1 = baseline_values.quantile(0.25)
        q3 = baseline_values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (new_values < lower_bound) | (new_values > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_rate = outlier_count / len(new_values)

        return {
            "column": column,
            "outlier_count": outlier_count,
            "outlier_rate": float(outlier_rate),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    def compute_all_metrics(
        self,
        new_df: pd.DataFrame,
        predictions: np.ndarray = None,
        actuals: np.ndarray = None,
    ) -> dict:
        """
        Compute all monitoring metrics.

        Returns a dictionary that can be printed, saved as JSON, or used by a drift detector.
        """
        results = {
            "null_rates": self.metric_3_null_rates(new_df),
            "ks_test": self.metric_4_ks_test(new_df),
            "psi": self.metric_5_psi(new_df),
            "data_freshness": self.metric_7_data_freshness(new_df),
            "duplicate_rate": self.metric_8_duplicate_rate(new_df),
            "outlier_rate": self.metric_9_outlier_rate(new_df),
        }

        if predictions is not None:
            results["prediction_distribution"] = self.metric_6_prediction_distribution(predictions)

        if predictions is not None and actuals is not None:
            results["accuracy"] = self.metric_1_accuracy(new_df, predictions, actuals)
            results["accuracy_by_zone"] = self.metric_2_accuracy_by_zone(new_df, predictions, actuals)

        return results


def save_metrics(metrics: dict, output_path: str = "monitoring_metrics.json") -> None:
    """Save metrics to a JSON file."""
    Path(output_path).write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    print("MetricComputer is ready. Import this class from your monitoring workflow.")
