from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "models/tuned_random_forest.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    """Root endpoint to confirm the API is running."""
    return "Customer Churn Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to make predictions."""
    try:
        # Parse the JSON request body
        data = request.get_json()

        # Check if all required fields are present
        required_fields = [
            "CreditScore", "Geography_Germany", "Geography_Spain", "Gender_Male",
            "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", 
            "IsActiveMember", "EstimatedSalary"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract features for prediction
        features = [
            data["CreditScore"], 
            data["Geography_Germany"], 
            data["Geography_Spain"], 
            data["Gender_Male"], 
            data["Age"], 
            data["Tenure"], 
            data["Balance"], 
            data["NumOfProducts"], 
            data["HasCrCard"], 
            data["IsActiveMember"], 
            data["EstimatedSalary"]
        ]

        # Make prediction
        prediction = model.predict([features])[0]

        # Return the result
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
