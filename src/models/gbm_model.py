from sklearn.ensemble import GradientBoostingRegressor

def train_gbm(X, y,i,j, weights):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.01*i,
        max_depth=j
    )
    model.fit(X, y, sample_weight=weights)
    return model