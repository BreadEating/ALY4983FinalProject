from pathlib import Path
import json
import pandas as pd

INP = Path("data/staged/data.csv")
REPORT = Path("artifacts/validation_report.json")
REPORT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INP)
    errors, warnings = [], []

    if "target" not in df.columns:
        errors.append("Missing 'target' column")
    else:
        if df["target"].isna().any():
            errors.append("'target' contains NaNs")
        uniq = set(df["target"].unique())
        if not uniq.issubset({0, 1}):
            errors.append(f"'target' has non-binary values: {sorted(uniq)}")

    if len(df) < 10000:
        warnings.append(f"Row count is low: {len(df)} (expected >= 10000)")

    missing = df.isna().sum().to_dict()

    report = {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "missing_per_column": missing
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"[validate] passed={report['passed']} -> {REPORT}")

    if not report["passed"]:
        raise SystemExit("[validate] failed. See artifacts/validation_report.json")

if __name__ == "__main__":
    main()