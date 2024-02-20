import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("TeePublic_review.csv", encoding='ISO-8859-1')

df = df.dropna()  # Drop rows with missing values for simplicity

column_list = ['store_location', 'year', 'title', 'review', 'review-label']
label_encoder = LabelEncoder()
for col in column_list:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop('review-label', axis=1)
y = df['review-label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(n_jobs=-1)
rf_model = RandomForestClassifier(n_jobs=-1)

xgb_training_time = []
xgb_precision = []
xgb_recall = []
xgb_f1 = []
xgb_accuracy = []

rf_training_time = []
rf_precision = []
rf_recall = []
rf_f1 = []
rf_accuracy = []

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
    'criterion': ['gini', 'entropy'],
}

print("Starded grid search for RF")
# Grid search for RandomForest
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
start_time = time.time()
grid_search_rf.fit(X_train, y_train)
end_time = time.time()
best_params_rf = grid_search_rf.best_params_

print("grid search time for RF:", end_time-start_time)

print("Starded grid search for XGBoost")
# Grid search for XGBoost
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy')
start_time = time.time()
grid_search_xgb.fit(X_train, y_train)
end_time = time.time()
best_params_xgb = grid_search_xgb.best_params_

print("grid search time for RF:", end_time-start_time)

print("grid search has ended")
print("best parameters for each model")
print("random forest:", best_params_rf)
print("XGBoost:", best_params_xgb)

runs = list(range(1, 10))

for i in runs:
    best_model_rf = RandomForestClassifier(**best_params_rf,n_jobs=-1)
    start_time_rf = time.time()
    best_model_rf.fit(X_train, y_train)
    end_time_rf = time.time()

    y_pred_rf = best_model_rf.predict(X_test)

    rf_training_time.append(end_time_rf - start_time_rf)
    rf_precision.append(precision_score(y_test, y_pred_rf, average='micro'))
    rf_recall.append(recall_score(y_test, y_pred_rf, average='micro'))
    rf_f1.append(f1_score(y_test, y_pred_rf, average='micro'))
    rf_accuracy.append(accuracy_score(y_test, y_pred_rf))

    best_model_xgb = XGBClassifier(**best_params_xgb,n_jobs=-1)
    start_time_xgb = time.time()
    best_model_xgb.fit(X_train, y_train)
    end_time_xgb = time.time()

    y_pred_xgb = best_model_xgb.predict(X_test)

    xgb_training_time.append(end_time_xgb - start_time_xgb)
    xgb_precision.append(precision_score(y_test, y_pred_xgb, average='micro'))
    xgb_recall.append(recall_score(y_test, y_pred_xgb, average='micro'))
    xgb_f1.append(f1_score(y_test, y_pred_xgb, average='micro'))
    xgb_accuracy.append(accuracy_score(y_test, y_pred_xgb))
    print("Current run:", i)

plt.plot(runs, rf_training_time, marker='o')
plt.xlabel("Run")
plt.ylabel("Training time")
plt.title("Training time For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_training_time, marker='o')
plt.xlabel("Run")
plt.ylabel("Training time")
plt.title("Training time For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_precision, marker='o')
plt.xlabel("Run")
plt.ylabel("precision")
plt.title("precision For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_precision, marker='o')
plt.xlabel("Run")
plt.ylabel("precision")
plt.title("precision For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_recall, marker='o')
plt.xlabel("Run")
plt.ylabel("Recall")
plt.title("Recall For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_recall, marker='o')
plt.xlabel("Run")
plt.ylabel("Recall")
plt.title("Recall For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_f1, marker='o')
plt.xlabel("Run")
plt.ylabel("F1 score")
plt.title("F1 score For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_f1, marker='o')
plt.xlabel("Run")
plt.ylabel("F1 score")
plt.title("F1 score For XGBoost")
plt.grid(True)
plt.show()

plt.plot(runs, rf_accuracy, marker='o')
plt.xlabel("Run")
plt.ylabel("accuracy")
plt.title("accuracy For Random Forest")
plt.grid(True)
plt.show()

plt.plot(runs, xgb_accuracy, marker='o')
plt.xlabel("Run")
plt.ylabel("accuracy")
plt.title("accuracy For XGBoost")
plt.grid(True)
plt.show()
