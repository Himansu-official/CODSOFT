import joblib
import pandas as pd
import os

def test_model(model_path, input_features, feature_names):
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Create a DataFrame for the input features
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    # Predict
    prediction = model.predict(input_df)
    return prediction

if __name__ == "__main__":
    # Absolute path for model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../models/random_forest.pkl")
    
    # Test features and feature names
    test_features = [600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1]
    feature_names = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", 
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]
    
    # Test the model
    prediction = test_model(model_path, test_features, feature_names)
    result = "Exited" if prediction[0] == 1 else "Not Exited"
    print(f"Prediction: {result}")
