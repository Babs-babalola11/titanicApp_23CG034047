from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        try:
            # Collect user inputs
            pclass = int(request.form["Pclass"])
            sex = request.form["Sex"]
            age = float(request.form["Age"])
            sibsp = int(request.form["SibSp"])
            parch = int(request.form["Parch"])
            fare = float(request.form["Fare"])
            embarked = request.form["Embarked"]

            # Create DataFrame
            df = pd.DataFrame({
                'Pclass': [pclass],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Sex_male': [1 if sex == 'male' else 0],
                'Embarked_Q': [1 if embarked == 'Q' else 0],
                'Embarked_S': [1 if embarked == 'S' else 0]
            })

            # Scale features
            df_scaled = scaler.transform(df)

            # Predict
            result = model.predict(df_scaled)[0]
            prediction_text = "Survived" if result == 1 else "Did Not Survive"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
