from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

def preprocess_data(file_path):
    """
    Preprocess the data and apply SMOTE to balance the training set.

    Parameters:
        file_path (str): Path to the dataset CSV file.

    Returns:
        X_train_balanced, X_test, y_train_balanced, y_test: Processed training and testing sets.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Encode categorical features
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

    # Drop unnecessary columns
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Split features and target
    X = data.drop('Exited', axis=1)
    y = data['Exited']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to balance the training set
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test

if __name__ == "__main__":
    # Test the preprocess_data function
    file_path = "data/Churn_Modelling.csv"  # Update this path if necessary

    try:
        # Call the preprocess function
        X_train_balanced, X_test, y_train_balanced, y_test = preprocess_data(file_path)

        # Print details about the processed data
        print("Preprocessing successful!")
        print(f"X_train_balanced shape: {X_train_balanced.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train_balanced distribution: {y_train_balanced.value_counts().to_dict()}")
        print(f"y_test distribution: {y_test.value_counts().to_dict()}")
    except Exception as e:
        print(f"An error occurred: {e}")
