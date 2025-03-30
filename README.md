# Handwritten-Digit-Classification-with-Neural-Network


## 📌 Project Overview

This project focuses on classifying handwritten digits using a neural network. The model is trained on the MNIST dataset, which consists of grayscale images of digits (0-9) with a resolution of 28x28 pixels. The goal is to build an efficient and accurate deep learning model to recognize digits with high precision.


## 📊 Dataset Details

Source: MNIST dataset (provided by Keras)

Training Data: 60,000 images

Testing Data: 10,000 images

Image Size: 28x28 pixels

Channels: Grayscale (1 channel)


## 🛠 Methodology

### 1.Data Preprocessing

Normalization of pixel values to the range [0,1].

Splitting data into training and testing sets.

### 2.Model Architecture

A fully connected neural network is used.

The model consists of the following layers:

Input layer (Flatten): Converts 28x28 matrix into a 1D vector.

Hidden layer (Dense): 128 neurons with ReLU activation.

Output layer (Dense): 10 neurons (one for each digit) with softmax activation.

### 3.Training and Evaluation

The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function.

Performance metrics include accuracy and loss.

The trained model is evaluated on the test dataset.

## 🎯 Results & Key Findings

Accuracy Achieved: Approximately 98% on test data.

Confusion Matrix Analysis:

The model performs well across all digits, with minimal misclassification.

Some confusion occurs between visually similar digits (e.g., 3 and 8, 4 and 9).

Key Observations:

Increasing hidden layers and neurons improves accuracy.

Training for more epochs can further optimize the model.

Using convolutional layers (CNN) instead of a fully connected network can enhance performance.

## 🏁 Conclusion

This project successfully demonstrates handwritten digit classification using a simple neural network. The high accuracy achieved highlights the effectiveness of deep learning in image recognition tasks. Future improvements could include experimenting with CNNs, data augmentation, and hyperparameter tuning to further enhance performance.
