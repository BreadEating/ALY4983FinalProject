import pandas as pd

def test_ingest_output_has_target():
    df = pd.read_csv("data/staged/data.csv")
    assert "target" in df.columns and len(df) > 0