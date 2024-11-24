import numpy as np
from tensorflow.keras.models import load_model

# Load the test data
x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Reshape x_test to include channel dimension
x_test = x_test.reshape(x_test.shape[0], 128, 32, 1)

# Load the trained model
model = load_model("results/models/handwritten_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
