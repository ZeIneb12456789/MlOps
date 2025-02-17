from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model and encoders
model = joblib.load("trained_model.joblib")
encoders = joblib.load("encoders.pkl")

app = FastAPI()

# Define the expected input schema
class InputData(BaseModel):
    State: str
    Account_length: int
    Area_code: int
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Encode categorical features
        try:
            state_encoded = encoders["State"].transform([data.State])[0].item()
            intl_plan_encoded = encoders["International plan"].transform([data.International_plan])[0].item()
            voicemail_encoded = encoders["Voice mail plan"].transform([data.Voice_mail_plan])[0].item()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

        # Prepare input data for the model
        input_data = np.array([[
            state_encoded, data.Account_length, data.Area_code,
            intl_plan_encoded, voicemail_encoded, data.Number_vmail_messages,
            data.Total_day_minutes, data.Total_day_calls, data.Total_day_charge,
            data.Total_eve_minutes, data.Total_eve_calls, data.Total_eve_charge,
            data.Total_night_minutes, data.Total_night_calls, data.Total_night_charge,
            data.Total_intl_minutes, data.Total_intl_calls, data.Total_intl_charge,
            data.Customer_service_calls
        ]], dtype=np.float32)  # Ensuring correct format

        # Debugging: Print input data shape
        print(f"Input Data Shape: {input_data.shape}")  # Should be (1, 19)

        # Ensure the model is compatible with the input
        if model.n_features_in_ != input_data.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Model expects {model.n_features_in_} features, but received {input_data.shape[1]}"
            )

        # Make prediction
        prediction = model.predict(input_data)[0].item()

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


