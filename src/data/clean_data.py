import pandas as pd

def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()
    df = df[df["avg_monthly_price"] > 0]

    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
    df = df.set_index("date").reindex(full_range)

    df["avg_monthly_price"] = df["avg_monthly_price"].interpolate()

    df = df.reset_index().rename(columns={"index": "date"})

    return df