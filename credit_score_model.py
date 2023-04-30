import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def train_and_save_model():
    data = pd.read_csv("data.csv")
    features = data[["income", "late_payments", "credit_utilization", "employment_years", "age", "credit_accounts", "debt_to_income"]]
    target = data["credit_score"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Save the trained model
    joblib.dump(model, "credit_score_model.joblib")

def load_model():
    model = joblib.load("credit_score_model.joblib")
    return model

# Uncomment the following line to train and save the model
train_and_save_model()

model = load_model()

def predict_credit_score(income, late_payments, credit_utilization, employment_years, age, credit_accounts, debt_to_income):
    prediction = model.predict([[income, late_payments, credit_utilization, employment_years, age, credit_accounts, debt_to_income]])
    return prediction[0]
