import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import warnings
import numpy as np

warnings.filterwarnings("ignore")

df = pd.read_csv("glass.csv", sep=",")

df = df.dropna()  # Drop rows with missing values for simplicity

# Set the style of seaborn
sns.set(style="whitegrid")

# Get numeric columns (excluding 'share_borders')
numeric_columns = df.select_dtypes(include=['number']).columns

# Calculate the number of rows and columns for subplots
num_rows = len(numeric_columns) // 2 + len(numeric_columns) % 2
num_cols = 2

# Create a figure with a specified size
plt.figure(figsize=(10, num_rows * 2))

# Iterate through each numeric column and create a boxplot
for i, column in enumerate(numeric_columns):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(y=column, data=df)
    plt.title(f'Boxplot of {column}')

# Adjust layout for better readability
plt.tight_layout()

# Show the plot
plt.show()

column_list = [
    "Type"
]

label_encoder = LabelEncoder()
for col in column_list:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop('Type', axis=1)
y = df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()

# Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
}

grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')

grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters and their corresponding accuracy
best_params_rf = grid_search_rf.best_params_
best_accuracy_rf = grid_search_rf.best_score_

best_model_rf = RandomForestClassifier(**best_params_rf)

start_time_rf = time.time()
best_model_rf.fit(X_train, y_train)
end_time_rf = time.time()

training_time_rf = end_time_rf - start_time_rf

y_pred_rf = best_model_rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='micro')
rf_recall = recall_score(y_test, y_pred_rf, average='micro')
rf_f1 = f1_score(y_test, y_pred_rf, average='micro')

print("Random Forest Metrics:")
print("Best Hyperparameters:", best_params_rf)
print("Accuracy:", best_accuracy_rf)
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

# GXBoost
# Define and train XGBoost model
xgb_model = XGBClassifier()

param_grid_xgb = {
    'learning_rate': [0.1, 0.2, 0.01],
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 7, 8],
    'sampling_method': ['uniform', 'gradient_based'],
    'subsample': [0.5, 1, 0.8],
    'colsample_bytree': [0.8, 1.0],
}

grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy')

grid_search_xgb.fit(X_train, y_train)

best_params_xgb = grid_search_xgb.best_params_
best_accuracy_xgb = grid_search_xgb.best_score_

best_model_xgb = XGBClassifier(**best_params_xgb)

start_time_xgb = time.time()
best_model_xgb.fit(X_train, y_train)
end_time_xgb = time.time()

xgb_training_time = end_time_xgb - start_time_xgb

y_pred_xgb = best_model_xgb.predict(X_test)

# Evaluate the classifier (using accuracy as an example)
xgb_precision = precision_score(y_test, y_pred_xgb, average='micro')
xgb_recall = recall_score(y_test, y_pred_xgb, average='micro')
xgb_f1 = f1_score(y_test, y_pred_xgb, average='micro')
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Metrics:")
print("Best Hyperparameters:", best_params_xgb)
print("Accuracy:", best_accuracy_xgb)
print('Accuracy: ', xgb_accuracy)
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1 Score:", xgb_f1)

# Results data
models = ['Random Forest', 'XGBoost']
precision_scores = [rf_precision, xgb_precision]
recall_scores = [rf_recall, xgb_recall]
f1_scores = [rf_f1, xgb_f1]

# Create a DataFrame for better plotting
results_df = pd.DataFrame(
    {'Model': models, 'Precision': precision_scores, 'Recall': recall_scores, 'F1 Score': f1_scores})

# Plotting
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Bar plot for Precision
sns.barplot(x='Model', y='Precision', data=results_df, ax=axes[0])
axes[0].set_title('Precision Scores')

# Bar plot for Recall
sns.barplot(x='Model', y='Recall', data=results_df, ax=axes[1])
axes[1].set_title('Recall Scores')

# Bar plot for F1 Score
sns.barplot(x='Model', y='F1 Score', data=results_df, ax=axes[2])
axes[2].set_title('F1 Scores')

plt.tight_layout()
plt.show()

# Training time comparison
print("\nTraining Time:")
print("Random Forest:", training_time_rf, "seconds")
print("XGBoost:", xgb_training_time, "seconds")

# Bar plot for Training Time
training_times = [training_time_rf, xgb_training_time]
sns.barplot(x=models, y=training_times)
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.show()