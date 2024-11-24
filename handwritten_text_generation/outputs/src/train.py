import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model import create_model
from sklearn.preprocessing import LabelEncoder

# Load data
x_train = np.load("data/processed/x_train.npy")
x_test = np.load("data/processed/x_test.npy")

# Load labels
y_train = np.genfromtxt("data/processed/y_train.csv", delimiter=",", dtype=str, skip_header=1)
y_test = np.genfromtxt("data/processed/y_test.csv", delimiter=",", dtype=str, skip_header=1)

# Extract labels
y_train = [label.split(",")[1] if "," in label else label for label in y_train]
y_test = [label.split(",")[1] if "," in label else label for label in y_test]

# Encode labels
combined_labels = np.concatenate((y_train, y_test))
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# One-hot encode labels
num_classes = len(label_encoder.classes_)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Reshape data
x_train = x_train.reshape((-1, 128, 32, 1))
x_test = x_test.reshape((-1, 128, 32, 1))

# Debugging: Check data shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Create model
input_shape = (128, 32, 1)
model = create_model(input_shape, num_classes)

# Debugging: Model summary
model.summary()

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Save model
model.save("results/models/handwritten_model.h5")

print("Model trained and saved successfully!")
