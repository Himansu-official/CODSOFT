import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess_images(image_dir, target_size=(128, 32)):
    """
    Preprocess images: resize, normalize, and extract labels from filenames.

    Args:
        image_dir (str): Directory containing raw image files.
        target_size (tuple): Size to which images will be resized (Height, Width).

    Returns:
        images (np.ndarray): Array of preprocessed images.
        labels (list): List of labels corresponding to images.
    """
    images = []
    labels = []

    # Debugging: Check directory
    print(f"Processing images from directory: {image_dir}")
    print(f"Files in directory: {os.listdir(image_dir)}")

    for file_name in os.listdir(image_dir):
        # Check for supported file formats (.png, .jpg, etc.)
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_dir, file_name)

            # Read and preprocess the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.resize(img, target_size)  # Resize to target size
            img = img / 255.0  # Normalize pixel values

            # Append image and label
            images.append(img)
            labels.append(file_name.split(".")[0])  # Assuming label is part of the file name

    # Convert to numpy arrays
    images = np.array(images)
    return images, labels

def save_preprocessed_data(images, labels, split_ratio=0.8):
    """
    Save preprocessed images and labels to files after splitting into train and test sets.

    Args:
        images (np.ndarray): Array of images.
        labels (list): List of labels corresponding to images.
        split_ratio (float): Ratio of train data to the total data.
    """
    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(labels)}")
    
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No data to save. Ensure images and labels are loaded correctly.")

    # One-hot encode labels
    unique_labels = sorted(list(set(labels)))
    print(f"Unique labels: {unique_labels}")
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    one_hot_labels = np.array([to_categorical(label_to_index[label], num_classes=len(unique_labels)) for label in labels])

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        images, one_hot_labels, test_size=1 - split_ratio, random_state=42
    )

    # Create processed data directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save data
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("Preprocessed data saved successfully.")

def main():
    """
    Main function to preprocess data and save processed files.
    """
    # Adjust image directory path
    image_dir = os.path.join(RAW_DATA_DIR, "images")

    # Preprocess images
    images, labels = preprocess_images(image_dir)

    # Save preprocessed data
    save_preprocessed_data(images, labels)

if __name__ == "__main__":
    main()
