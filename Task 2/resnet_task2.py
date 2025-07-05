import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

# Parameters
IMAGE_SIZE = (112,112)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = 'Task 2/dataset_combined'

# Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

num_classes = train_gen.num_classes
input_shape = IMAGE_SIZE + (3,)

# Model with additional layers
def create_model():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save Model
model.save("Task 2/resnet_softmax_model.h5")

# Evaluate - Training Set
train_gen.reset()
train_preds = []
train_labels = []
train_probs = []

for i in range(len(train_gen)):
    x_batch, y_batch = train_gen[i]
    preds = model.predict(x_batch, verbose=0)
    train_probs.extend(preds)
    train_preds.extend(np.argmax(preds, axis=1))
    train_labels.extend(y_batch)

# Evaluate - Validation Set
val_gen.reset()
val_preds = []
val_labels = []
val_probs = []

for i in range(len(val_gen)):
    x_batch, y_batch = val_gen[i]
    preds = model.predict(x_batch, verbose=0)
    val_probs.extend(preds)
    val_preds.extend(np.argmax(preds, axis=1))
    val_labels.extend(y_batch)

# Classification Reports - Print and Save
train_report = classification_report(train_labels, train_preds)
print("\n[✓] Training Classification Report:")
print(train_report)
with open('Task 2/train_classification_report.txt', 'w') as f:
    f.write(train_report)
print("[✓] Training classification report saved as train_classification_report.txt")

val_report = classification_report(val_labels, val_preds)
print("\n[✓] Validation Classification Report:")
print(val_report)
with open('Task 2/val_classification_report.txt', 'w') as f:
    f.write(val_report)
print("[✓] Validation classification report saved as val_classification_report.txt")

# Plot Accuracy and Loss
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# ROC Curve
plt.subplot(1, 3, 3)
train_labels_bin = label_binarize(train_labels, classes=range(num_classes))
val_labels_bin = label_binarize(val_labels, classes=range(num_classes))
train_probs = np.array(train_probs)
val_probs = np.array(val_probs)

for i in range(num_classes):
    fpr_t, tpr_t, _ = roc_curve(train_labels_bin[:, i], train_probs[:, i])
    fpr_v, tpr_v, _ = roc_curve(val_labels_bin[:, i], val_probs[:, i])
    roc_auc_t = auc(fpr_t, tpr_t)
    roc_auc_v = auc(fpr_v, tpr_v)
    plt.plot(fpr_t, tpr_t, label=f'Train Class {i} (AUC = {roc_auc_t:.2f})')
    plt.plot(fpr_v, tpr_v, '--', label=f'Val Class {i} (AUC = {roc_auc_v:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('Task 2/training_metrics.png')
print("[✓] Plots saved as training_metrics.png")
plt.show()

