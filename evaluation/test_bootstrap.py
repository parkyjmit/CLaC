#!/usr/bin/env python3
"""
Test script for bootstrapping confidence interval calculation.

This tests the bootstrap_confidence_interval function with known data
to verify it produces reasonable results.
"""

import numpy as np
import sys
sys.path.append('.')

from compute_metrics import bootstrap_confidence_interval


def test_bootstrap_basic():
    """Test basic bootstrapping with known accuracy."""
    print("="*80)
    print("TEST 1: Basic Bootstrapping")
    print("="*80)

    # Simulate 1000 test samples with 75% accuracy
    np.random.seed(42)
    n_samples = 1000
    true_accuracy = 0.75
    outcomes = np.random.binomial(1, true_accuracy, n_samples)

    print(f"Simulated data: {n_samples} samples, true accuracy = {true_accuracy}")
    print(f"Observed accuracy: {outcomes.mean():.4f}")

    # Compute bootstrap CI
    result = bootstrap_confidence_interval(outcomes, n_bootstrap=1000)

    print(f"\nBootstrap Results:")
    print(f"  Mean:     {result['mean']:.4f}")
    print(f"  Std:      {result['std']:.4f}")
    print(f"  95% CI:   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    # Check if true accuracy is within CI
    in_ci = result['ci_lower'] <= true_accuracy <= result['ci_upper']
    print(f"\nTrue accuracy ({true_accuracy}) within 95% CI: {in_ci}")

    # For large sample size, std should be approximately sqrt(p*(1-p)/n)
    theoretical_std = np.sqrt(true_accuracy * (1 - true_accuracy) / n_samples)
    print(f"\nTheoretical SE: {theoretical_std:.4f}")
    print(f"Bootstrap std:  {result['std']:.4f}")
    print(f"Ratio:          {result['std'] / theoretical_std:.2f} (should be ~1.0)")

    assert in_ci, "True accuracy should be within 95% CI"
    assert 0.8 < result['std'] / theoretical_std < 1.2, "Bootstrap std should match theoretical SE"
    print("\n✓ TEST 1 PASSED\n")


def test_bootstrap_small_sample():
    """Test bootstrapping with small sample size."""
    print("="*80)
    print("TEST 2: Small Sample Size")
    print("="*80)

    # Small sample: 50 samples with 80% accuracy
    np.random.seed(123)
    n_samples = 50
    true_accuracy = 0.80
    outcomes = np.random.binomial(1, true_accuracy, n_samples)

    print(f"Simulated data: {n_samples} samples, true accuracy = {true_accuracy}")
    print(f"Observed accuracy: {outcomes.mean():.4f}")

    result = bootstrap_confidence_interval(outcomes, n_bootstrap=1000)

    print(f"\nBootstrap Results:")
    print(f"  Mean:     {result['mean']:.4f}")
    print(f"  Std:      {result['std']:.4f}")
    print(f"  95% CI:   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    # CI should be wider for small samples
    ci_width = result['ci_upper'] - result['ci_lower']
    print(f"\nCI width: {ci_width:.4f}")
    print(f"Std:      {result['std']:.4f}")

    assert ci_width > 0.05, "CI should be reasonably wide for small sample"
    print("\n✓ TEST 2 PASSED\n")


def test_bootstrap_perfect_accuracy():
    """Test bootstrapping with perfect accuracy."""
    print("="*80)
    print("TEST 3: Perfect Accuracy (100%)")
    print("="*80)

    # All correct
    outcomes = np.ones(100)

    print(f"Simulated data: 100 samples, all correct")

    result = bootstrap_confidence_interval(outcomes, n_bootstrap=1000)

    print(f"\nBootstrap Results:")
    print(f"  Mean:     {result['mean']:.4f}")
    print(f"  Std:      {result['std']:.4f}")
    print(f"  95% CI:   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    # For perfect accuracy, all bootstrap samples should also be 1.0
    assert result['mean'] == 1.0, "Mean should be exactly 1.0"
    assert result['std'] == 0.0, "Std should be 0.0 for perfect accuracy"
    assert result['ci_lower'] == 1.0 and result['ci_upper'] == 1.0, "CI should be [1.0, 1.0]"

    print("\n✓ TEST 3 PASSED\n")


def test_bootstrap_zero_accuracy():
    """Test bootstrapping with zero accuracy."""
    print("="*80)
    print("TEST 4: Zero Accuracy (0%)")
    print("="*80)

    # All incorrect
    outcomes = np.zeros(100)

    print(f"Simulated data: 100 samples, all incorrect")

    result = bootstrap_confidence_interval(outcomes, n_bootstrap=1000)

    print(f"\nBootstrap Results:")
    print(f"  Mean:     {result['mean']:.4f}")
    print(f"  Std:      {result['std']:.4f}")
    print(f"  95% CI:   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    assert result['mean'] == 0.0, "Mean should be exactly 0.0"
    assert result['std'] == 0.0, "Std should be 0.0 for zero accuracy"
    assert result['ci_lower'] == 0.0 and result['ci_upper'] == 0.0, "CI should be [0.0, 0.0]"

    print("\n✓ TEST 4 PASSED\n")


def test_bootstrap_reproducibility():
    """Test that bootstrapping with same seed produces same results."""
    print("="*80)
    print("TEST 5: Reproducibility")
    print("="*80)

    np.random.seed(999)
    outcomes = np.random.binomial(1, 0.7, 200)

    print(f"Running bootstrapping twice with same data...")

    # First run
    result1 = bootstrap_confidence_interval(outcomes, n_bootstrap=500)

    # Second run (should be identical due to fixed seed in bootstrap function)
    result2 = bootstrap_confidence_interval(outcomes, n_bootstrap=500)

    print(f"\nRun 1: Mean={result1['mean']:.6f}, CI=[{result1['ci_lower']:.6f}, {result1['ci_upper']:.6f}]")
    print(f"Run 2: Mean={result2['mean']:.6f}, CI=[{result2['ci_lower']:.6f}, {result2['ci_upper']:.6f}]")

    assert result1['mean'] == result2['mean'], "Mean should be identical"
    assert result1['ci_lower'] == result2['ci_lower'], "CI lower should be identical"
    assert result1['ci_upper'] == result2['ci_upper'], "CI upper should be identical"

    print("\n✓ TEST 5 PASSED\n")


def test_bootstrap_empty():
    """Test bootstrapping with empty data."""
    print("="*80)
    print("TEST 6: Empty Data")
    print("="*80)

    outcomes = []

    print(f"Testing with empty list...")

    result = bootstrap_confidence_interval(outcomes, n_bootstrap=1000)

    print(f"\nBootstrap Results:")
    print(f"  Mean:     {result['mean']}")
    print(f"  Std:      {result['std']}")
    print(f"  CI:       [{result['ci_lower']}, {result['ci_upper']}]")

    assert np.isnan(result['mean']), "Mean should be NaN for empty data"
    assert np.isnan(result['std']), "Std should be NaN for empty data"

    print("\n✓ TEST 6 PASSED\n")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BOOTSTRAPPING CONFIDENCE INTERVAL - TEST SUITE")
    print("="*80 + "\n")

    try:
        test_bootstrap_basic()
        test_bootstrap_small_sample()
        test_bootstrap_perfect_accuracy()
        test_bootstrap_zero_accuracy()
        test_bootstrap_reproducibility()
        test_bootstrap_empty()

        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe bootstrap_confidence_interval function is working correctly!")
        print("You can now use it for zero-shot QA and retrieval evaluations.\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
