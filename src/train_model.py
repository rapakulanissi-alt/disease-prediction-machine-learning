import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

data = pd.read_csv("data/disease_data.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = data.drop("disease", axis=1)
y = data["disease"]

print("\nSymptoms used by the model:")
print(list(X.columns))

# -----------------------------------------
# 3. Split the dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------------------
# 4. Create the Machine Learning model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -----------------------------------------
# 5. Train the model
# -----------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# -----------------------------------------
# 6. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------
# 7. Calculate accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# -----------------------------------------
# 8. Classification report
# -----------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------------------
# 9. Save the trained model
# -----------------------------------------

joblib.dump(model, "models/disease_model.pkl")

print("\nModel saved successfully!")
print("File: models/disease_model.pkl")
