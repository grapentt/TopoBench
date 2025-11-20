#!/usr/bin/env python3
"""Simple coverage check script."""

import subprocess
import sys

# Run tests with coverage using subprocess to avoid import issues
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--source=topobench/data/preprocessor/ondisk_inductive.py",
        "-m",
        "pytest",
        "test/data/preprocessor/test_ondisk_inductive.py",
        "-v",
    ],
    capture_output=True,
    text=True,
)

print(result.stdout)
print(result.stderr)

# Show coverage report
result2 = subprocess.run(
    [sys.executable, "-m", "coverage", "report"],
    capture_output=True,
    text=True,
)

print("\n" + "=" * 80)
print("COVERAGE REPORT")
print("=" * 80)
print(result2.stdout)

# Extract coverage percentage
for line in result2.stdout.split("\n"):
    if "ondisk_inductive" in line:
        parts = line.split()
        if len(parts) >= 4:
            coverage_pct = parts[-1].rstrip("%")
            print(f"\n✓ Coverage: {coverage_pct}%")
            if int(coverage_pct) >= 93:
                print("✓ MEETS 93% REQUIREMENT!")
            else:
                print(f"✗ Below 93% requirement (need {93 - int(coverage_pct)}% more)")
