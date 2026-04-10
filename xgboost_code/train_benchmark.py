import numpy as np
import pandas as pd

from xgboost_scratch import XGBoost

df = pd.read_csv(r"C:/Users/CG-DTE/Desktop/xgboost_implementation/data/california_housing.csv")

X = df.drop(columns=["MedHouseVal"]).values
y = df["MedHouseVal"].values


# Initializing Model
model = XGBoost(n_estimators=10,learning_rate=0.1,max_depth=3)


# Training Model
model.fit(X, y)

# Predictions
predictions = model.predict(X)


# Evaluation (RMSE)
rmse = np.sqrt(np.mean((y - predictions) ** 2))

#print("RMSE:", rmse)

#BENCHMARKING My implementation vs Real xgboost rmse

from xgboost import XGBRegressor

# Train real XGBoost
xgb = XGBRegressor(n_estimators=10,max_depth=3,learning_rate=0.1)

xgb.fit(X, y)

xgb_preds = xgb.predict(X)

xgb_rmse = np.sqrt(np.mean((y - xgb_preds) ** 2))

print("My XGBoost RMSE:", rmse)
print("Real XGBoost RMSE:", xgb_rmse)