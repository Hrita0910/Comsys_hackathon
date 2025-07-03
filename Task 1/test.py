import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, ttk
import mediapipe as mp
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('Task 1/gender_classifier_task1.h5')

# Image preprocessing function
def preprocess_image(image_path, image_size=(224, 224)):
    """Preprocess image for model prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict gender and confidence
def predict_gender(image_path):
    """Predict gender and return label with confidence score."""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]  # Sigmoid output
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = 'Male' if prediction > 0.5 else 'Female'
    return label, confidence * 100  # Confidence as percentage

# Function to draw rectangle around face using MediaPipe
def draw_face_rectangle(image_path):
    """Detect face using MediaPipe and draw a rectangle around it."""
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    # Read image with OpenCV for MediaPipe processing
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(img_rgb)
    
    # Convert to PIL Image for drawing
    pil_img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    
    if results.detections:
        # Use the first detected face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        img_height, img_width = img.shape[:2]
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * img_width)
        y = int(bbox.ymin * img_height)
        w = int(bbox.width * img_width)
        h = int(bbox.height * img_height)
        
        # Draw rectangle (red, width=3)
        draw.rectangle((x, y, x + w, y + h), outline='red', width=3)
    
    # Release MediaPipe resources
    face_detection.close()
    
    return pil_img

# Tkinter GUI class
class GenderPredictorApp:
    def __init__(self, root):
        """Initialize the Tkinter GUI."""
        self.root = root
        self.root.title("Gender Predictor")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")

        # Styling
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("TLabel", font=("Helvetica", 14), background="#f0f0f0")

        # Title
        self.title_label = ttk.Label(root, text="Gender Classification", style="TLabel")
        self.title_label.pack(pady=10)

        # Upload button
        self.upload_button = ttk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Image display label
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)

        # Result label
        self.result_label = ttk.Label(root, text="Prediction: None", style="TLabel")
        self.result_label.pack(pady=10)

    def upload_image(self):
        """Handle image upload, face detection, and prediction."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                # Draw rectangle around face
                img_with_rectangle = draw_face_rectangle(file_path)
                
                # Resize for display
                img_display = img_with_rectangle.resize((200, 200))
                img_tk = ImageTk.PhotoImage(img_display)
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk  # Keep reference

                # Predict gender
                label, confidence = predict_gender(file_path)
                self.result_label.configure(
                    text=f"Prediction: {label}, Confidence: {confidence:.2f}%"
                )
            except Exception as e:
                self.result_label.configure(text=f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = GenderPredictorApp(root)
    root.mainloop()