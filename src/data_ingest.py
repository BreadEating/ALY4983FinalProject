from pathlib import Path
import pandas as pd

RAW = Path("data/raw/UCI_Credit_Card.csv")
OUT = Path("data/staged/data.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW)
    # Drop ID and rename target
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.rename(columns={"default.payment.next.month": "target"})
    df.to_csv(OUT, index=False)
    print(f"[ingest] rows={len(df)}, cols={len(df.columns)} -> {OUT}")

if __name__ == "__main__":
    main()