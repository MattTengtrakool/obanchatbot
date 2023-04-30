from flask import Flask, render_template, request, jsonify
import numpy as np
import credit_score_model as cs_model
import pandas as pd

app = Flask(__name__)
model = cs_model.load_model()  # Make sure you've implemented the `load_model` function in your `credit_score_model.py` file

@app.route("/")
def index():
    return render_template("index.html")

providers_df = pd.read_csv("providers.csv")
loan_providers = providers_df.to_dict(orient="records")

@app.route('/predict_credit_score', methods=['POST'])
def predict_credit_score():
    income = float(request.form['income'])
    late_payments = int(request.form['late_payments'])
    credit_utilization = float(request.form['credit_utilization'])
    employment_years = int(request.form['employment_years'])
    age = int(request.form['age'])
    credit_accounts = int(request.form['credit_accounts'])
    debt_to_income = float(request.form['debt_to_income'])

    features = np.array([income, late_payments, credit_utilization, employment_years, age, credit_accounts, debt_to_income]).reshape(1, -1)
    credit_score = model.predict(features)[0]

    suitable_providers = [
        provider for provider in loan_providers
        if (income >= provider["min_income"] and
            debt_to_income <= provider["max_debt_to_income"] and
            credit_utilization >= provider["min_credit_utilization"] and
            credit_utilization <= provider["max_credit_utilization"] and
            employment_years >= provider["min_employment_years"])
    ]

    return jsonify({"credit_score": credit_score, "suitable_providers": suitable_providers})
    
if __name__ == '__main__':
    app.run(debug=True)
