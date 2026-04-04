import numpy as np
from sklearn.metrics import mean_squared_error
from src.config.config import load_config
from src.models.xgb_model import train_xgb
from src.models.gbm_model import train_gbm

def residual_model(X_train, y_train, X_test, y_test, weights):
    
    def evaluate_model(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    # Track best model
    best_rmse = float("inf")
    best_residual_model = None
    best_model_name = None
    best_params = {}

    for i in range(1, 10):
        for j in range(1, 10):
    
            # XGBoost
            xgb_model = train_xgb(X_train, y_train, i, j, weights)
            
            xgb_pred = xgb_model.predict(X_test)
            xgb_rmse = evaluate_model(y_test, xgb_pred)
    
            if xgb_rmse < best_rmse:
                best_rmse = xgb_rmse
                best_residual_model = xgb_model
                best_model_name = "XGBoost"
                best_params = {"learning_rate": 0.01*i, "max_depth": j}
    
            # Gradient Boosting
            gbm_model = train_gbm(X_train, y_train, i, j, weights)
            gbm_pred = gbm_model.predict(X_test)
            gbm_rmse = evaluate_model(y_test, gbm_pred)
    
            if gbm_rmse < best_rmse:
                best_rmse = gbm_rmse
                best_residual_model = gbm_model
                best_model_name = "Gradient Boosting"
                best_params = {"learning_rate": 0.01*i, "max_depth": j}

    reports = {
        'residual model': {
            "Model": best_model_name,
            "Best RMSE": round(best_rmse, 3),
            "Best Parameters": best_params
        }
    }
    return (best_residual_model, reports['residual model'])