import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/disease_model.pkl")

print("====================================")
print("       DISEASE PREDICTION SYSTEM")
print("====================================")

print("\nEnter 1 for YES and 0 for NO.\n")

# Get symptoms from the user
fever = int(input("Do you have fever? (1/0): "))
cough = int(input("Do you have cough? (1/0): "))
headache = int(input("Do you have headache? (1/0): "))
fatigue = int(input("Do you have fatigue? (1/0): "))
sore_throat = int(input("Do you have sore throat? (1/0): "))
runny_nose = int(input("Do you have runny nose? (1/0): "))
vomiting = int(input("Do you have vomiting? (1/0): "))
diarrhea = int(input("Do you have diarrhea? (1/0): "))
joint_pain = int(input("Do you have joint pain? (1/0): "))

# Create input data
input_data = pd.DataFrame([{
    "fever": fever,
    "cough": cough,
    "headache": headache,
    "fatigue": fatigue,
    "sore_throat": sore_throat,
    "runny_nose": runny_nose,
    "vomiting": vomiting,
    "diarrhea": diarrhea,
    "joint_pain": joint_pain
}])

# Make prediction
prediction = model.predict(input_data)

print("\n====================================")
print("Predicted Disease:", prediction[0])
print("====================================")
print("\nNote: This is an educational ML project,")
print("not a medical diagnosis.")
