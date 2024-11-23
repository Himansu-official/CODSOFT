import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from scripts.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nEvaluating Logistic Regression...")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, predictions):.2f}")
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\nRunning Grid Search for Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    grid_search_rf = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid_rf,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1,
    )
    grid_search_rf.fit(X_train, y_train)
    best_model = grid_search_rf.best_estimator_
    print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

    print("\nEvaluating Tuned Random Forest...")
    predictions = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"Tuned Random Forest ROC-AUC Score: {roc_auc_score(y_test, predictions):.2f}")

    # Visualize Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # Feature Importance
    feature_importances = best_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances)
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    return best_model

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\nRunning Grid Search for XGBoost...")
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0],
    }
    grid_search_xgb = GridSearchCV(
        estimator=XGBClassifier(),
        param_grid=param_grid_xgb,
        cv=3,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1,
    )
    grid_search_xgb.fit(X_train, y_train)
    best_model = grid_search_xgb.best_estimator_
    print(f"Best parameters for XGBoost: {grid_search_xgb.best_params_}")

    print("\nEvaluating Tuned XGBoost...")
    predictions = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"Tuned XGBoost ROC-AUC Score: {roc_auc_score(y_test, predictions):.2f}")

    # Visualize Feature Importance
    feature_importances = best_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances)
    plt.title("Feature Importance - XGBoost")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    return best_model

def main():
    # Preprocess the data
    file_path = "data/Churn_Modelling.csv"
    X_train_balanced, X_test, y_train_balanced, y_test = preprocess_data(file_path)

    # Train models
    logistic_model = train_logistic_regression(X_train_balanced, y_train_balanced, X_test, y_test)
    random_forest_model = train_random_forest(X_train_balanced, y_train_balanced, X_test, y_test)
    xgboost_model = train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)

    # Save the best models
    print("\nSaving Models...")
    joblib.dump(logistic_model, "models/logistic_model.pkl")
    print("Logistic Regression model saved successfully!")
    joblib.dump(random_forest_model, "models/tuned_random_forest.pkl")
    print("Tuned Random Forest model saved successfully!")
    joblib.dump(xgboost_model, "models/tuned_xgboost.pkl")
    print("Tuned XGBoost model saved successfully!")

if __name__ == "__main__":
    main()
