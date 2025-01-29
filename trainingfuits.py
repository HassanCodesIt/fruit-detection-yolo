import os
import cv2
import multiprocessing

# Set the start method for multiprocessing (optional, can help on Windows)
multiprocessing.set_start_method('spawn', force=True)

# Ensure OpenCV uses a single thread (optional, may help with data loading)
cv2.setNumThreads(0)

# Train the YOLOv8 model with reduced workers
from ultralytics import YOLO

# Load the YOLOv8 model (replace with your model if needed)
model = YOLO('yolov8n.pt')  # You can use any YOLOv8 model

# Define your desired directory for saving the model
custom_save_dir = "C:\Users\\hassa\\OneDrive\\Desktop\\project\\trained_models"

# Train the model and specify the project and name
results = model.train(
    data="C:/Users/hassa/.cache/kagglehub/datasets/lakshaytyagi01/fruit-detection/versions/1/Fruits-detection/data.yaml",  # Path to your dataset configuration file
    epochs=10,                 # Number of epochs
    batch=8,                   # Batch size
    imgsz=640,                 # Image size (resize images to 640x640)
    workers=0,                 # Number of workers for data loading
    project=custom_save_dir,   # Specify your desired directory
    name="fruit_detection"     # Name of the experiment folder
)

# Move the final weights to a specific directory (optional)
trained_weights = os.path.join(custom_save_dir, "fruit_detection", "weights", "best.pt")  # Path to best weights
final_save_path = os.path.join(custom_save_dir, "fruit_detection_best.pt")

if os.path.exists(trained_weights):
    os.rename(trained_weights, final_save_path)
    print(f"Model saved to {final_save_path}")
else:
    print("Trained weights not found.")
