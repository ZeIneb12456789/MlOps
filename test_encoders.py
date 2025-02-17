import joblib

# Load the encoders
encoders = joblib.load("encoders.pkl")

# Print the keys and an example transformation
print("Keys in encoders.pkl:", encoders.keys())

# Check if categorical feature encoders exist
for key in encoders.keys():
    print(f"Encoder for {key}: {type(encoders[key])}")

