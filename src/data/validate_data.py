def validate_data(df):
    assert "date" in df.columns
    assert "avg_monthly_price" in df.columns
    assert df["avg_monthly_price"].isnull().sum() == 0