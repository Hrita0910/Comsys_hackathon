# Comsys_hackathon

## Task 1 : Gender Classification

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
- ✅ Open VS Code(do not create a folder). Open terminal and git clone this project. The particular folder will be created on your computer. Open the required folder (Comsys_hackathon).
- ✅ Go to the Task 1 folder (cd '.\Task 1\')
- ✅ pip install -r requirements.txt - Install all the libraries
- ✅ Go back to the prevoius folder (cd ..)
- ✅ Now run, python '.\Task 1\task1.py' - Run and train the model
- ✅ Finally, python '.\Task 1\test.py' - Test on custom images


