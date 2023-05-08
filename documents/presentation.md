# Skin Cancer Classification with CNN and Dash App Deployment | University of Zurich

**A CNN Model for Skin Cancer Classification**

**Authors:** Roman Stadler, Carolyne Huang, Rahel Eberle and Manuel Pfister

**Date:** 2023-05-01

**Version:** 1.0

# Model

## Introduction
In this project, we aim to classify skin cancer using a Convolutional Neural Network (CNN) model. The dataset we are using is available at [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), which consists of images of different types of skin cancer. The dataset has been preprocessed and augmented before training the CNN model. Finally, the trained model is used to build a Dash app for deployment.

## Data Preprocessing
The preprocessing steps include reading the metadata, resizing the images to a uniform size (64x64), and converting the images from BGR to RGB. The images are then converted to a NumPy array and normalized by dividing by 255.0. Labels are encoded using a mapping dictionary and then one-hot encoded.

## Data Structure
The data is organized in the following manner:
- X: Contains image data
- y: Contains corresponding labels

These are then split into training and test sets using the train_test_split function from sklearn, with 80% data for training and 20% data for testing.

## Data Augmentation
Data augmentation is performed using the ImageDataGenerator function from Keras. It includes rotation, width and height shift, zoom, and horizontal and vertical flipping to increase the training data's variability.

## CNN Model
The CNN model consists of the following layers:
1. 3 Conv2D layers with increasing filters (32, 64, and 128), each followed by BatchNormalization, MaxPooling2D, and Dropout layers.
2. A Flatten layer to convert the 3D output to 1D.
3. A Dense layer with 512 units and ReLU activation, followed by BatchNormalization and Dropout.
4. A final Dense layer with softmax activation for multi-class classification.

The model is compiled with the Adam optimizer and categorical_crossentropy loss.

## Model Training
The model is trained using the fit method with data augmentation for 50 epochs and a batch size of 32. The validation data is the test set, which is not augmented.

## Model Evaluation
The model is evaluated on the test set to obtain the test loss and accuracy.

# Dash App Deployment
After training and evaluation, the model is saved as an H5 file, which can be used for deploying the model in a Dash app. The label_mapping JSON file is also saved to map labels to their original names in the app.

In conclusion, this project showcases the process of training a CNN model for skin cancer classification, from data preprocessing and augmentation to model training, evaluation, and deployment using a Dash app.

## Introduction
Skin cancer classification using a Convolutional Neural Network (CNN) model. This project involves loading and using a trained model for predictions and implementing a user interface using Dash.

## Importing Libraries
Suppress TensorFlow warnings and import required libraries and modules, including Dash and TensorFlow.

## Loading Trained Model and Label Mapping
Load the trained CNN model from the specified path and load the label mapping file (JSON).

## Dash App Layout
Define the app layout using Dash Bootstrap Components. Display the University of Zurich logo and the title. Show a table of lesion types and their abbreviations. Provide buttons to open the camera and capture images. Allow users to upload images in JPG format. Display the uploaded image and prediction results.

## Image Processing and Prediction
Define a function to process the uploaded image and make a prediction. Resize and convert the image to an array. Make a prediction using the CNN model. Get the predicted class, probability, and label.

## Callbacks and Interactivity
Create Dash app callbacks to handle displaying the uploaded image, opening the camera, storing the captured image, and making predictions. Prevent updates when no input is provided.

## Running the App
Set the server variable and run the Dash app with the specified settings.






