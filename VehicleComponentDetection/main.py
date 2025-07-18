from threading import Thread
import cv2
import numpy as np
import time
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy
from rtsp_server import create_rtsp_server
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # Import Ultralytics Annotator

# Define the class names for detection
class_names = ['back_glass', 'back_left_side_light', 'back_right_side_light', 'door', 
               'front_bumper', 'front_glass', 'front_left_side_door', 'front_left_side_light', 
               'front_left_side_mirror', 'front_right_side_door', 'front_right_side_light', 
               'front_right_side_mirror', 'hood', 'rear_bumper', 'wheel']

# Set confidence threshold
conf_threshold = 0.4

# Initialize video source and output
input = videoSource('csi://0', options={'width': 1280, 'height': 720, 'flipMethod': 'rotate-180'})
output = videoOutput("rtsp://0.0.0.0:1234")

# Start RTSP server in a separate thread
Thread(target=create_rtsp_server, daemon=True).start()

# Load YOLO model - use 'best.pt' or any other model you have
# For testing, we'll use yolov8n.pt which is a small model
try:
    # Try to load your custom model first
    model_path = 'best.pt'
    model = YOLO(model_path)
    print(f"Loaded custom model: {model_path}")
except Exception as e:
    # Fall back to a standard model if custom model fails to load
    print(f"Error loading custom model: {e}")
    model_path = 'yolov8n.pt'
    model = YOLO(model_path)
    print(f"Loaded standard model: {model_path}")

# For FPS calculation
prev_time = time.time()
fps = 0
inference_time = 0

# Function to process YOLO results and draw bounding boxes
def process_results(img, results):
    # Create an annotator for the image
    annotator = Annotator(img)
    
    # Process each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0]  # xyxy format (x1, y1, x2, y2)
            
            # Get class and confidence
            c = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Only process detections above confidence threshold
            if conf >= conf_threshold:
                # Get class name
                if c < len(class_names):
                    label = class_names[c]
                else:
                    label = f"class_{c}"
                
                # Create label with confidence
                text = f"{label} {conf:.2f}"
                
                # Draw box and label
                annotator.box_label(b, text, color=(0, 255, 0))
    
    # Return the annotated image
    return annotator.result()

# Function to run YOLO detection on a frame
def run_detection(frame):
    global inference_time
    
    # Start timing for inference
    inference_start = time.time()
    
    # Run YOLO detection
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    
    # Calculate inference time
    inference_time = (time.time() - inference_start) * 1000  # ms
    
    return results

try:
    # Capture frames until end-of-stream (or the user exits)
    while True:
        # Capture frame
        cuda_img = input.Capture(format='rgb8', timeout=1000)
        
        if cuda_img is None:  # if a timeout occurred
            continue
        
        # Convert CUDA image to numpy array for processing
        frame = cudaToNumpy(cuda_img)
        
        # Run YOLO detection on the frame
        results = run_detection(frame)
        
        # Process results and draw bounding boxes
        frame = process_results(frame, results)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS and inference time
        cv2.putText(frame, 'FPS: {:.1f}'.format(fps), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Inference: {:.1f}ms'.format(inference_time), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to CUDA image and render
        cuda_output = cudaFromNumpy(frame)
        output.Render(cuda_output)
        
        # Exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break

except KeyboardInterrupt:
    print('Exiting...')
except Exception as e:
    print('Error:', e)
finally:
    # Clean up resources
    print('Cleaning up resources...')
    del input
    del output