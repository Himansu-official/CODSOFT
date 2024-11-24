import tensorflow as tf
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # CNN Encoder
    cnn = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Flatten()(cnn)

    # Fully Connected Layers
    dense = layers.Dense(128, activation="relu")(cnn)
    outputs = layers.Dense(num_classes, activation="softmax")(dense)

    # Final Model
    model = tf.keras.Model(inputs, outputs)
    return model

