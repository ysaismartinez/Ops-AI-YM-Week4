"""
Lightweight tests for the monitoring framework.

Expected repo location:
    week4/tests/test_monitoring.py

Run from the repository root with:
    pytest week4/tests/test_monitoring.py

These tests intentionally use small synthetic datasets so they are fast and easy
to understand in a homework submission.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow this test file to import week4/scripts/metric_template.py when the test
# lives under week4/tests/.
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if SCRIPT_DIR.exists():
    sys.path.insert(0, str(SCRIPT_DIR))

from metric_template import MetricComputer


def make_baseline_df() -> pd.DataFrame:
    """Create a small healthy baseline dataset."""
    return pd.DataFrame(
        {
            "zone_id": [1, 1, 2, 2, 3, 3],
            "trip_count": [10, 11, 12, 13, 14, 15],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
        }
    )


def make_new_df() -> pd.DataFrame:
    """Create a small new dataset with a shifted trip_count distribution."""
    return pd.DataFrame(
        {
            "zone_id": [1, 1, 2, 2, 3, 3],
            "trip_count": [30, 31, 32, 33, 34, 35],
            "timestamp": pd.date_range("2024-02-01", periods=6, freq="h"),
        }
    )


def test_overall_accuracy_metric() -> None:
    baseline_df = make_baseline_df()
    new_df = make_new_df()
    computer = MetricComputer(baseline_df)

    predictions = np.array([1, 1, 0, 1, 0, 0])
    actuals = np.array([1, 0, 0, 1, 1, 0])

    accuracy = computer.metric_1_accuracy(new_df, predictions, actuals)

    assert accuracy == 4 / 6


def test_accuracy_by_zone_metric() -> None:
    baseline_df = make_baseline_df()
    new_df = make_new_df()
    computer = MetricComputer(baseline_df)

    predictions = np.array([1, 1, 0, 1, 0, 0])
    actuals = np.array([1, 0, 0, 1, 1, 0])

    accuracy_by_zone = computer.metric_2_accuracy_by_zone(new_df, predictions, actuals)

    assert accuracy_by_zone["1"] == 0.5
    assert accuracy_by_zone["2"] == 1.0
    assert accuracy_by_zone["3"] == 0.5


def test_ks_test_detects_distribution_shift() -> None:
    baseline_df = make_baseline_df()
    new_df = make_new_df()
    computer = MetricComputer(baseline_df)

    result = computer.metric_4_ks_test(new_df)

    assert result["column"] == "trip_count"
    assert result["statistic"] > 0
    assert result["p_value"] < 0.05
    assert result["drift_detected"] is True


def test_psi_increases_when_distribution_changes() -> None:
    baseline_df = make_baseline_df()
    new_df = make_new_df()
    computer = MetricComputer(baseline_df)

    psi = computer.metric_5_psi(new_df)

    assert psi > 0.20


def test_duplicate_rate_metric() -> None:
    baseline_df = make_baseline_df()
    new_df = pd.concat([make_new_df(), make_new_df().iloc[[0]]], ignore_index=True)
    computer = MetricComputer(baseline_df)

    result = computer.metric_8_duplicate_rate(new_df)

    assert result["duplicate_count"] == 1
    assert result["duplicate_rate"] == 1 / 7


def test_compute_all_metrics_returns_core_sections() -> None:
    baseline_df = make_baseline_df()
    new_df = make_new_df()
    computer = MetricComputer(baseline_df)

    results = computer.compute_all_metrics(new_df)

    assert "null_rates" in results
    assert "ks_test" in results
    assert "psi" in results
    assert "duplicate_rate" in results
    assert "outlier_rate" in results
