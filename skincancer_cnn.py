# Project: Skin Cancer Classification
# Description: This file contains the code to train the CNN model for skin cancer classification.
# Author: Roman Stadler, Carolyne Huang, Rahel Eberle and Manuel Pfister
# Date: 2023-05-01
# License: MIT License
# Version: 1.0
# ======================================================================================================================

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ======================================================================================================================
# 1. Data Preparation
# Path to the unzipped dataset
data_dir = 'www/data/'

# Path to the metadata file
metadata_path = os.path.join(data_dir, 'metadata.csv')

# Read metadata
metadata = pd.read_csv(metadata_path)

# Set image size
image_size = 64

# Prepare data and labels
X = []
y = []

image_files = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
image_ids = [os.path.basename(image_path)[:-4] for image_path in image_files]

for image_id in image_ids:
    image_path = os.path.join(data_dir, 'images', image_id + '.jpg')
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    X.append(image)
    
    if image_id in metadata['image_id'].values:
        y.append(metadata.loc[metadata['image_id'] == image_id, 'dx'].values[0])
    else:
        y.append('others')

# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)


# Normalize image data
X /= 255.0

# Label encoding
label_mapping = {label: index for index, label in enumerate(np.unique(y))}

# Save label_mapping to a JSON file
label_mapping_path = 'www/label_mapping.json'
with open(label_mapping_path, 'w') as f:
    json.dump(label_mapping, f)

y = np.array([label_mapping[label] for label in y])

# One-hot encoding for labels
y = to_categorical(y, num_classes=len(label_mapping))

# ======================================================================================================================
# 2. Data Augmentation
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# ======================================================================================================================
# 3. Model Definition
# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the CNN model
input_shape = (image_size, image_size, 3)
num_classes = len(label_mapping)

model = create_cnn_model(input_shape, num_classes)
model.summary()

# ======================================================================================================================
# 4. Model Training
# Train the model
batch_size = 32
epochs = 10
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs)

# ======================================================================================================================   
# 5. Model Evaluation
# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ======================================================================================================================
# Save the trained model to a file
model.save('www/model/skin_cnn_model.h5')
