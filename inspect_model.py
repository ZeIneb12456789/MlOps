import joblib
import pickle

# Load the encoders from the pickle file
with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Inspect each label encoder's classes_
for column, le in label_encoders.item():
    print(f"Classes for {column}: {le.classes_}")

# Example: Check how a specific encoder works (for the 'State' column)
state_encoder = label_encoders["State"]
print("Encoded value for 'CA':", state_encoder.transform(["CA"]))
print("Decoded value for encoded value 0:", state_encoder.inverse_transform([0]))

# Load the trained model
model = joblib.load("trained_model.joblib")

# Check the feature names expected by the model
if hasattr(model, "feature_names_in_"):
    print("Model expects the following features:")
    print(model.feature_names_in_)
else:
    print("The model does not store feature names. Check the training code.")
