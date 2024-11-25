import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(file_path):
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Load the data
    data = pd.read_csv(file_path)
    # Preprocess the data
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    return X, y

def train_model(data_path, model_path):
    # Preprocess the data
    X, y = preprocess_data(data_path)

    # Address class imbalance using SMOTE
    print("Applying SMOTE to address class imbalance...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"Original dataset size: {X.shape}, Resampled dataset size: {X_res.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Construct dynamic paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Churn_Modelling.csv")
    model_path = os.path.join(script_dir, "../models/random_forest.pkl")
    train_model(data_path, model_path)
