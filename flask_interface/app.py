from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace with the container name of your model service
MODEL_API_URL = "http://model_service:5000/predict"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = {
            "State": request.form["State"],
            "Account_length": int(request.form["Account_length"]),
            "Area_code": int(request.form["Area_code"]),
            "International_plan": request.form["International_plan"],
            "Voice_mail_plan": request.form["Voice_mail_plan"],
            "Number_vmail_messages": int(request.form["Number_vmail_messages"]),
            "Total_day_minutes": float(request.form["Total_day_minutes"]),
            "Total_day_calls": int(request.form["Total_day_calls"]),
            "Total_day_charge": float(request.form["Total_day_charge"]),
            "Total_eve_minutes": float(request.form["Total_eve_minutes"]),
            "Total_eve_calls": int(request.form["Total_eve_calls"]),
            "Total_eve_charge": float(request.form["Total_eve_charge"]),
            "Total_night_minutes": float(request.form["Total_night_minutes"]),
            "Total_night_calls": int(request.form["Total_night_calls"]),
            "Total_night_charge": float(request.form["Total_night_charge"]),
            "Total_intl_minutes": float(request.form["Total_intl_minutes"]),
            "Total_intl_calls": int(request.form["Total_intl_calls"]),
            "Total_intl_charge": float(request.form["Total_intl_charge"]),
            "Customer_service_calls": int(request.form["Customer_service_calls"])
        }

        response = requests.post(MODEL_API_URL, json=user_input)
        result = response.json()
        return render_template("index.html", prediction=result.get("prediction", "Error"))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

