from ultralytics import YOLO
import os
import cv2

# Path to the model file
model_path = r'C:\Users\hassa\OneDrive\Desktop\project\trained_models\fruit_detection_best.pt'

# Verify the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path.")
    exit(1)

print("Model file found. Loading the model...")

# Load the YOLO model
model = YOLO(model_path)

# Path to the video file
video_path = r'C:\Users\hassa\OneDrive\Desktop\project\fruitsvideo.mp4'

# Verify the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}. Please check the path.")
    exit(1)

print("Video file found. Running predictions...")

# Load the video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {frame_count} frames")

# Directory to save results
save_dir = r'C:\Users\hassa\OneDrive\Desktop\project\results'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Output video path
output_video_path = os.path.join(save_dir, "annotated_fruitsvideo.mp4")

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when the video ends

    # Run the YOLO model on the current frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

print(f"Annotated video saved at: {output_video_path}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
