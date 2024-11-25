import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)
    data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    X = data.drop('Exited', axis=1)
    y = data['Exited']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y
