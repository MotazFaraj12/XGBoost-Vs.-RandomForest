import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import warnings
import numpy as np
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

df = pd.read_csv("possum.csv", sep=",")

df = df.dropna()  # Drop rows with missing values for simplicity

encode_c = ["Pop", "sex"]

label_encoder = LabelEncoder()
for col in encode_c:
    df[col] = label_encoder.fit_transform(df[col])

ouliers = ["case", "site", "Pop", "sex", "age", "hdlngth", "skullw", "totlngth", "taill", "footlgth", "earconch", "eye",
           "chest", "belly"]

for c in ouliers:
    # Compute percentiles using Pandas quantile() function
    percentile_25 = df[c].quantile(0.25)
    percentile_50 = df[c].quantile(0.5)
    percentile_75 = df[c].quantile(0.75)

    # Compute interquartile range (IQR)
    iqr = percentile_75 - percentile_25

    LowerBound_charges = percentile_25 - 1.5 * iqr
    UpperBound_charges = percentile_75 + 1.5 * iqr

    NumRecordsBefore = df.shape[0]
    DroppedRecords = df[(df[c] < LowerBound_charges) | (df[c] > UpperBound_charges)].shape[0]

    df[c] = np.where(df[c] > UpperBound_charges, UpperBound_charges,
                     np.where(df[c] < LowerBound_charges, LowerBound_charges,
                              df[c]))

X = df.drop('age', axis=1)
y = df['age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_jobs=-1, random_state=42)
rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)

xgb_training_time = []
xgb_mse = []
xgb_mae = []
xgb_r2 = []
bp_xgb = []

rf_training_time = []
rf_mse = []
rf_mae = []
rf_r2 = []
bp_rf = []

param_grid_xgb = {
    'learning_rate': [0.1, 0.2, 0.01],
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 7, 8],
    'sampling_method': ['uniform', 'gradient_based'],
    'subsample': [0.5, 1, 0.8],
    'colsample_bytree': [0.8, 1.0],
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'absolute_error'],
}


runs = list(range(1, 13))
for i in runs:
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("Started grid search for RF")
    # Grid search for RandomForest
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
    start_time = time.time()
    grid_search_rf.fit(X_train_pca, y_train)
    end_time = time.time()
    best_params_rf = grid_search_rf.best_params_
    bp_rf.append(best_params_rf)

    print("grid search time for RF:", end_time - start_time)

    print("Started grid search for XGBoost")
    # Grid search for XGBoost
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
    start_time = time.time()
    grid_search_xgb.fit(X_train_pca, y_train)
    end_time = time.time()
    best_params_xgb = grid_search_xgb.best_params_
    bp_xgb.append(best_params_xgb)

    print("grid search time for XGBoost:", end_time - start_time)

    print("grid search has ended")
    print("best parameters for each model")
    print("random forest:", best_params_rf)
    print("XGBoost:", best_params_xgb)

    best_model_rf = RandomForestRegressor(**best_params_rf, n_jobs=-1, random_state=42)
    start_time_rf = time.time()
    best_model_rf.fit(X_train_pca, y_train)
    end_time_rf = time.time()

    y_pred_rf = best_model_rf.predict(X_test_pca)

    rf_training_time.append(end_time_rf - start_time_rf)
    rf_mse.append(mean_squared_error(y_test, y_pred_rf))
    rf_mae.append(mean_absolute_error(y_test, y_pred_rf))
    rf_r2.append(r2_score(y_test, y_pred_rf))

    best_model_xgb = XGBRegressor(**best_params_xgb, n_jobs=-1, random_state=42)
    start_time_xgb = time.time()
    best_model_xgb.fit(X_train_pca, y_train)
    end_time_xgb = time.time()

    y_pred_xgb = best_model_xgb.predict(X_test_pca)

    xgb_training_time.append(end_time_xgb - start_time_xgb)
    xgb_mse.append(mean_squared_error(y_test, y_pred_xgb))
    xgb_mae.append(mean_absolute_error(y_test, y_pred_xgb))
    xgb_r2.append(r2_score(y_test, y_pred_xgb))

    print("Current run:", i)
    print("XGBoost r2:", r2_score(y_test, y_pred_xgb))
    print("RF r2:", r2_score(y_test, y_pred_rf))
    print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
    print("RF MSE:", mean_squared_error(y_test, y_pred_rf))
    print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
    print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
    print("//////////////////////new run stared//////////////////////\n\n\n")

print("Random forest best parameter for each degree:", bp_rf)

print("Random forest best parameter for each degree:", bp_xgb)

print("Random forest MAE",)
print("XGBoost MAE",)

print("Random forest MSE",)
print("XGBoost MSE",)

print("Random forest R2",)
print("XGBoost R2",)


plt.plot(runs, rf_training_time, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Training time")
plt.title("Training time For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_training_time, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Training time")
plt.title("Training time For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_mse, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_mse, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_mae, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Mean Absolute Error")
plt.title("Mean Absolute Error For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_mae, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("Mean Absolute Error")
plt.title("Mean Absolute Error For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_r2, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("R-squared score")
plt.title("R-squared score For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_r2, marker='o')
plt.xlabel("Degrees of dimensionality")
plt.ylabel("R-squared score")
plt.title("R-squared score For XGBoost")
plt.grid(True)
plt.show()
