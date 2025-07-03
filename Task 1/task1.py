import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Step 1: Data Preprocessing
def load_and_preprocess_data(data_dir, image_size=(224, 224), batch_size=32):
    """Load and preprocess dataset from directory."""
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0,1]
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    
    # Only rescaling for validation and test
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_ds = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',  # Binary labels: 0 (female), 1 (male)
        shuffle=True
    )
    
    # Load validation data
    val_ds = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_ds, val_ds

# Step 2: Define the CNN Model
def create_cnn_model(input_shape=(224, 224, 3)):
    """Create a simple CNN model for gender classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),  # Regularization to prevent overfitting
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.75),  # Regularization to prevent overfitting
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # Adjusted learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Step 3: Evaluate the Model
def evaluate_model(model, data_generator, dataset_name='validation'):
    """Evaluate model on validation/test data and return metrics."""
    # Get predictions and true labels
    predictions = []
    true_labels = []
    pred_probs = []
    
    for images, labels in data_generator:
        preds = model.predict(images, verbose=0)
        pred_probs.extend(preds.flatten())
        predictions.extend((preds > 0.35).astype(int).flatten())  
        true_labels.extend(labels.astype(int))
        if len(predictions) >= data_generator.samples:
            break
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    report = classification_report(true_labels, predictions, target_names=['Female', 'Male'])
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'Task 1/{dataset_name}_confusion_matrix.png')
    plt.close()
    
    # Compute and plot ROC curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'Task 1/{dataset_name}_roc_curve.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'roc_auc': roc_auc
    }

# Step 4: Test Function for Submission
def test_model(model_path, test_data_path, image_size=(224, 224), batch_size=32):
    """Load model and evaluate on test data."""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_ds = test_datagen.flow_from_directory(
        test_data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_ds, dataset_name='test')
    return metrics

# Main execution
if __name__ == '__main__':
    # Dataset path (adjust as needed)
    data_dir = 'Task 1/data/Comsys/Comsys_Hackathon5/Task_A'
    
    # Load and preprocess data
    train_ds, val_ds = load_and_preprocess_data(data_dir)
    
    # Create and train model
    model = create_cnn_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3.5,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=35,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on training set
    train_metrics = evaluate_model(model, train_ds, dataset_name='training')
    print("\nTraining Accuracy: {:.4f}".format(train_metrics['accuracy']))
    print("Training ROC AUC: {:.4f}".format(train_metrics['roc_auc']))
    print("\nTraining Classification Report:\n", train_metrics['classification_report'])
    
    # Save training classification report
    with open('Task 1/training_classification_report(task1).txt', 'w') as f:
        f.write("Training Accuracy: {:.4f}\n".format(train_metrics['accuracy']))
        f.write("Training ROC AUC: {:.4f}\n\n".format(train_metrics['roc_auc']))
        f.write("Training Classification Report:\n")
        f.write(train_metrics['classification_report'])
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_ds, dataset_name='validation')
    print("\nValidation Accuracy: {:.4f}".format(val_metrics['accuracy']))
    print("Validation ROC AUC: {:.4f}".format(val_metrics['roc_auc']))
    print("\nValidation Classification Report:\n", val_metrics['classification_report'])
    
    # Save validation classification report
    with open('Task 1/validation_classification_report(task1).txt', 'w') as f:
        f.write("Validation Accuracy: {:.4f}\n".format(val_metrics['accuracy']))
        f.write("Validation ROC AUC: {:.4f}\n\n".format(val_metrics['roc_auc']))
        f.write("Validation Classification Report:\n")
        f.write(val_metrics['classification_report'])
    
    # Save model weights
    model.save('Task 1/gender_classifier_task1.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Task 1/training_history.png')
    plt.close()