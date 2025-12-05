import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load or simulate dataset
# ---------------------------
data = {
    'Pregnancies': [2, 4, 3, 1, 5, 3, 1, 2, 6, 3],
    'Glucose': [120, 150, 99, 180, 120, 115, 140, 130, 110, 140],
    'BloodPressure': [70, 88, 65, 90, 75, 80, 85, 78, 82, 90],
    'Insulin': [80, 130, 0, 150, 200, 120, 95, 105, 110, 125],
    'BMI': [28.1, 35.6, 23.4, 40.2, 28.9, 33.5, 30.0, 32.1, 29.4, 31.7],
    'DiabetesPedigreeFunction': [0.5, 1.2, 0.3, 0.8, 0.6, 0.9, 0.4, 0.7, 0.6, 0.5],
    'Age': [33, 45, 29, 60, 50, 40, 55, 38, 41, 47],
    'SmokingStatus': ['Never', 'Former', 'Current', 'Never', 'Former', 'Current', 'Never', 'Former', 'Current', 'Never'],
    'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# ---------------------------
# 2. Preprocessing
# ---------------------------
categorical_features = ['SmokingStatus']
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 3. CatBoost Model Training
# ---------------------------
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=0,
    random_seed=42
)

train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, cat_features=categorical_features)

model.fit(train_pool)

# ---------------------------
# 4. Prediction & Evaluation
# ---------------------------
y_pred = model.predict(test_pool)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# 5. Feature Importance (Optional)
# ---------------------------
feature_importance = model.get_feature_importance(prettified=True)

# ---------------------------
# 6. Input for Prediction
# ---------------------------

def predict_diabetes():
    print("\nEnter the following details to predict diabetes:")

    # Taking input values for each feature
    pregnancies = int(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    blood_pressure = float(input("BloodPressure: "))
    insulin = float(input("Insulin: "))
    weight = float(input("Weight (kg): "))  # Weight input
    height = float(input("Height (m): "))  # Height input
    age = int(input("Age: "))
    smoking_status = input("SmokingStatus (Never, Former, Current): ")

    # Input for family history of diabetes
    parents_with_diabetes = int(input("Number of parents with diabetes (0 or 1): "))
    siblings_with_diabetes = int(input("Number of siblings with diabetes (0 or 1): "))
    uncles_with_diabetes = int(input("Number of uncles with diabetes (0 or 1): "))

    # Calculate DiabetesPedigreeFunction based on family history
    diabetes_pedigree_function = parents_with_diabetes + siblings_with_diabetes + uncles_with_diabetes

    # Calculate BMI internally
    bmi = weight / (height ** 2)

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age],
        'SmokingStatus': [smoking_status]
    })

    # Predict using the trained model
    input_pool = Pool(input_data, cat_features=categorical_features)
    prediction = model.predict(input_pool)
    
    # Output the prediction
    if prediction == 0:
        print("\nPrediction: No Diabetes")
    else:
        print("\nPrediction: Diabetes")

# Run the prediction function
predict_diabetes()
print("Accuracy:", accuracy_score(y_test, y_pred))
