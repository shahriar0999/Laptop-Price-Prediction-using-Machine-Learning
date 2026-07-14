import json
import sys
from pathlib import Path

REPORT_FILE = Path("reports/evaluation_report.json")

if not REPORT_FILE.exists():
    print("❌ reports/evaluation_report.json not found.")
    sys.exit(1)

with open(REPORT_FILE, "r") as f:
    report = json.load(f)

# Expected thresholds
MIN_R2 = 0.85
MAX_OVERFIT_GAP = 0.05

checks = {
    "Threshold Status": report["threshold_status"] == "PASS",
    "R² Score": report["r2_log"] >= MIN_R2,
    "Cross Validation": report["cv_r2_mean"] >= MIN_R2,
    "Overfit Gap": report["overfit_gap"] <= MAX_OVERFIT_GAP,
}

print("=" * 50)
print("MODEL EVALUATION REPORT")
print("=" * 50)

print(f"Model           : {report['model']}")
print(f"R² Score        : {report['r2_log']:.4f}")
print(f"CV R²           : {report['cv_r2_mean']:.4f}")
print(f"Overfit Gap     : {report['overfit_gap']:.4f}")
print(f"Threshold Status: {report['threshold_status']}")
print()

failed = False

for check, passed in checks.items():
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {check}")

    if not passed:
        failed = True

print()

if failed:
    print("❌ Model quality validation failed.")
    sys.exit(1)

print("✅ Model quality validation passed.")
