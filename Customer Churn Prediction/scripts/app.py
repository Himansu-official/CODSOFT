from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/random_forest.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)
logging.info(f"Model loaded successfully from {model_path}")

# Expected feature names for input validation
EXPECTED_FEATURES = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

@app.route('/')
def home():
    """Default route for testing."""
    return "Welcome to the Customer Churn Prediction API! Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict customer churn using input features provided in JSON format.
    Supports both single and batch predictions.
    """
    try:
        # Parse input JSON
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Handle batch input or single input
        if isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = pd.DataFrame([input_data])

        # Validate input features
        missing_features = [feature for feature in EXPECTED_FEATURES if feature not in input_df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Ensure correct feature order
        input_df = input_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        # Make predictions
        predictions = model.predict(input_df)
        results = ["Exited" if pred == 1 else "Not Exited" for pred in predictions]

        # Return results
        return jsonify({"predictions": results})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
