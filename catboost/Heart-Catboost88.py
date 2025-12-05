import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

# Load dataset
df = pd.read_csv('Heart/heart.csv')  # Ensure correct path

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Optuna objective function
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 1),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "eval_metric": "Accuracy",
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
        "verbose": 0,
        "random_seed": 42
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80)  # Increase trials for better accuracy

# Final model with best parameters
best_params = study.best_params
best_params.update({
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "auto_class_weights": "Balanced",
    "verbose": 100,
    "random_seed": 42
})

# Train final model
model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")


# ----------------- Sample Prediction -------------------
# Sample input as a dictionary with feature names matching training data
sample_input = {
    'age': 63,
    'sex': 1,          
    'cp': 3,            
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 1,         
    'oldpeak': 2.3,    
    'slope': 0,
    'ca': 1,            
    'thal': 2         
}

input_df = pd.DataFrame([sample_input])

# Predict using CatBoost (trained on full features)
prediction = model.predict(input_df)

print("\nPrediction on Sample Input:")
print("Predicted Class:", prediction[0])
if prediction[0] == 0:
    print("\033[1mThe Person does NOT have Heart Disease\033[0m")
else:
    print("\033[1mThe Person HAS Heart Disease\033[0m")


# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            cmap='YlGnBu', xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("CatBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


