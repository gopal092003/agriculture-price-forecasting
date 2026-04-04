from xgboost import XGBRegressor

def train_xgb(X, y, i, j, weights):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01*i,
        max_depth=j,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y, sample_weight=weights)
    return model