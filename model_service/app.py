from flask import Flask, request, jsonify, abort
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("trained_model.joblib")
encoders = joblib.load("encoders.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Check if all required fields are present
        required_fields = [
            "State", "Account_length", "Area_code", "International_plan", "Voice_mail_plan",
            "Number_vmail_messages", "Total_day_minutes", "Total_day_calls", "Total_day_charge",
            "Total_eve_minutes", "Total_eve_calls", "Total_eve_charge",
            "Total_night_minutes", "Total_night_calls", "Total_night_charge",
            "Total_intl_minutes", "Total_intl_calls", "Total_intl_charge",
            "Customer_service_calls"
        ]
        
        for field in required_fields:
            if field not in data:
                abort(400, f"Missing field: {field}")

        # Encode categorical features
        try:
            state_encoded = encoders["State"].transform([data["State"]])[0].item()
            intl_plan_encoded = encoders["International plan"].transform([data["International_plan"]])[0].item()
            voicemail_encoded = encoders["Voice mail plan"].transform([data["Voice_mail_plan"]])[0].item()
        except ValueError as e:
            abort(400, f"Encoding error: {str(e)}")

        # Prepare input data for the model
        input_data = np.array([[  
            state_encoded, data["Account_length"], data["Area_code"],
            intl_plan_encoded, voicemail_encoded, data["Number_vmail_messages"],
            data["Total_day_minutes"], data["Total_day_calls"], data["Total_day_charge"],
            data["Total_eve_minutes"], data["Total_eve_calls"], data["Total_eve_charge"],
            data["Total_night_minutes"], data["Total_night_calls"], data["Total_night_charge"],
            data["Total_intl_minutes"], data["Total_intl_calls"], data["Total_intl_charge"],
            data["Customer_service_calls"]
        ]], dtype=np.float32)

        # Debugging: Print input data shape
        print(f"Input Data Shape: {input_data.shape}")  # Should be (1, 19)

        # Ensure model input compatibility
        if model.n_features_in_ != input_data.shape[1]:
            abort(400, f"Model expects {model.n_features_in_} features, but received {input_data.shape[1]}")

        # Make prediction
        prediction = model.predict(input_data)[0].item()

        return jsonify({"prediction": prediction})

    except Exception as e:
        abort(500, str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

