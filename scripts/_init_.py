from scripts.preprocess import preprocess_data

# Path to your dataset
file_path = "data/Churn_Modelling.csv"

# Run preprocessing and print results
try:
    X_train_balanced, X_test, y_train_balanced, y_test = preprocess_data(file_path)

    print("Preprocessing completed successfully!")
    print(f"Balanced training target distribution: {y_train_balanced.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
except Exception as e:
    print(f"An error occurred: {e}")