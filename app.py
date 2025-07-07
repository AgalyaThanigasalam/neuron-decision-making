from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model("neuron_model.keras")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [
            float(request.form["OverTime"]),
            float(request.form["DistanceFromHome"]),
            float(request.form["YearsSinceLastPromotion"]),
            float(request.form["JobInvolvement"]),
            float(request.form["WorkLifeBalance"]),
            float(request.form["MonthlyIncome"]),
            float(request.form["TotalWorkingYears"]),
            float(request.form["StockOptionLevel"]),
            float(request.form["YearsAtCompany"]),
            float(request.form["EnvironmentSatisfaction"]),
        ]
        scaled = scaler.transform([input_data])
        prediction = model.predict(scaled)[0][0]
        result = "Yes" if prediction > 0.5 else "No"
        return render_template("result.html", result=result, prob=round(prediction, 2))
    except:
        return "Error: Invalid input."

if __name__ == "__main__":
    app.run(debug=True)
