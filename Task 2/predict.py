import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
MODEL_PATH = 'Task 2/resnet_softmax_model.h5'
DATASET_DIR = 'Task 2/dataset_combined'
IMAGE_SIZE = (112, 112)
MIN_CONFIDENCE = 0.1  # Lowered threshold for demonstration (adjust as needed)
TTA_NUM = 5  # Number of test-time augmentations

# Load model
print("[✓] Loading model...")
try:
    model = load_model(MODEL_PATH)
    print(f"[✓] Model loaded from {MODEL_PATH}")
    print(f"[!] Model input shape: {model.input_shape}")
except Exception as e:
    print(f"[X] Failed to load model: {e}")
    exit()

# Get class names
try:
    class_names = sorted([d for d in os.listdir(DATASET_DIR) 
                         if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"[✓] Found {len(class_names)} classes in dataset")
except Exception as e:
    print(f"[X] Failed to read class directories: {e}")
    exit()

# Create test-time augmentation generator
tta_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

def predict_with_tta(face_img):
    """Make predictions with test-time augmentation"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face_img = cv2.resize(face_img, IMAGE_SIZE)
    
    preds = []
    for _ in range(TTA_NUM):
        # Apply random augmentation
        aug_img = tta_gen.random_transform(face_img)
        aug_img = aug_img / 255.0  # Normalize
        p = model.predict(np.expand_dims(aug_img, axis=0), verbose=0)
        preds.append(p)
    
    return np.mean(preds, axis=0)

def detect_and_predict(image_path):
    """Main prediction function"""
    # Load image
    print(f"\n[•] Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("[X] Failed to load image.")
        return
    
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("[X] No faces detected.")
        return

    for (x, y, w, h) in faces:
        face_img = orig[y:y+h, x:x+w]
        
        # Make prediction with TTA
        preds = predict_with_tta(face_img)
        preds = tf.nn.softmax(preds).numpy()  # Removed temperature scaling
        
        # Get top prediction
        predicted_index = np.argmax(preds)
        confidence = preds[0][predicted_index]
        label = class_names[predicted_index]
        
        # Print debug info
        print(f"\nTop predictions for face at ({x},{y})-{w}x{h}:")
        top5 = np.argsort(preds[0])[-5:][::-1]
        for i in top5:
            print(f"  {class_names[i]:<20}: {preds[0][i]:.2%}")
        
        # Always show the top prediction regardless of confidence
        cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(orig, f"{label} ({confidence:.2%})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if confidence > 0.5 else (0, 0, 255), 2)
    
    # Show result
    cv2.imshow("Prediction", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    test_images = [
        "Task 2/Hrithik_Roshan_0001.jpg" 
    ]
    
    for img in test_images:
        if os.path.exists(img):
            detect_and_predict(img)
        else:
            print(f"[X] File not found: {img}")