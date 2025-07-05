# Comsys_hackathon

## ✅ Task 1 : Gender Classification

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow to classify gender (male/female) based on facial images. It includes model training, evaluation (with ROC/Confusion matrix), and a real-time GUI application for image-based prediction using MediaPipe and Tkinter.

---

### Project Features

- ✅ Gender classification using CNN
- ✅ Data augmentation for improved generalization
- ✅ Performance evaluation with accuracy, precision, recall, F1-score, ROC curve, and confusion matrix
- ✅ Interactive GUI with face detection using MediaPipe
- ✅ Saves evaluation results and model checkpoints


### Model Overview

The CNN model consists of:
- ✅ 4 convolutional layers with increasing filter sizes
- ✅ MaxPooling and Dropout layers for regularization
- ✅ Final Dense layer with sigmoid activation for binary classification
- ✅ Early stopping with patience to avoid overfitting

### Steps to run the model

- ✅ Open VS Code(do not create a folder). Open terminal and clone this repository. The particular folder will be created on your computer. Open the required folder (Comsys_hackathon).
- ✅ Go to the Task 1 folder (cd '.\Task 1\')
- ✅ pip install -r requirements.txt - Install all the libraries
- ✅ Go back to the prevoius folder (cd ..)
- ✅ Now run, python '.\Task 1\task1.py' - Run and train the model
- ✅ Finally, python '.\Task 1\test.py' - Test on custom images


## ✅ Task 2 : Face Recognition

This project implements a CNN-based image classification model using ResNet50 (with fine-tuning) to classify celebrity faces from a multi-class dataset. It includes dataset merging, model training, evaluation (with classification report & ROC curves), and prediction on custom test images using face detection and test-time augmentation (TTA).

### Project Features

- ✅ Classifies faces using a fine-tuned ResNet50 model
- ✅ Merges training and validation datasets into a single combined dataset
- ✅ Data augmentation during training to improve generalization
- ✅ Evaluation with classification report, accuracy, loss plots, and multi-class ROC curves
- ✅ Face detection & prediction script with TTA and confidence scores
- ✅ Auto-saves classification reports and plots
- ✅ Test-time augmentation for more stable predictions

### Model Architecture

- ✅ A ResNet50 base model pretrained on ImageNet 
- ✅ A GlobalAveragePooling2D layer to reduce spatial dimensions
- ✅ A Dense layer with 512 units and ReLU activation for learning complex features
- ✅ A BatchNormalization and Dropout layer for regularization
- ✅ A final Dense layer with softmax activation to output class probabilities for multi-class classification

### Steps to run the model

- ✅ Open VS Code. Go to your terminal
- ✅ Clone the repository
- ✅ Navigate to the project directory (Comsys_hackathon). Go to Task 2 folder (cd '.\Task 2\')
- ✅ Go back to the prevoius folder (cd ..)
- ✅ Run, python '.\Task 2\count_images.py' - Counts the number of common images of val and train
- ✅ Now run, python '.\Task 2\dataset_combine.py' - Merge the train and val datasets into one
- ✅ Run the model, python '.\Task 2\task2.py' - Run and train the model
- ✅ Finally test the model, python '.\Task2\predict.py' - Test on some images


