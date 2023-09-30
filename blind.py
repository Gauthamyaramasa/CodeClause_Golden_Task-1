import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load labels from trainLabels19.csv
labels_df = pd.read_csv('trainLabels19.csv')

# Define paths to the image files in 'resized train 19' directory
image_dir = 'resized_train_19'
image_paths = [os.path.join(image_dir, filename + '.jpg') for filename in labels_df['id_code']]

# Load and preprocess images
def load_and_preprocess_image(image_path, label):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values
        return img, label
    except tf.errors.NotFoundError:
        print(f"Image not found: {image_path}")
        return None, None

# Load and preprocess images and labels
images_and_labels = [
    load_and_preprocess_image(image_path, label)
    for image_path, label in zip(image_paths, labels_df['diagnosis'])
]

# Filter out None values (skipped images)
images_and_labels = [(img, label) for img, label in images_and_labels if img is not None]

# Split data into images and labels
images, labels = zip(*images_and_labels)
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes for severity levels
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training
epochs = 10
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=epochs,
    verbose=1
)

# User input and prediction
def predict_blindness_severity(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (128, 128))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize pixel values
        img = tf.expand_dims(img, axis=0)  # Add batch dimension

        prediction = model.predict(img)
        severity_level = np.argmax(prediction)
        return severity_level
    except tf.errors.NotFoundError:
        print(f"Image not found: {image_path}")
        return None

if __name__ == "__main__":
    image_path = input("Enter the path to an eye image for severity prediction: ")
    severity_level = predict_blindness_severity(image_path)
    if severity_level is not None:
        severity_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        print(f"Predicted severity level: {severity_labels[severity_level]}")
