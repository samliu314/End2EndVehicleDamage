#!/usr/bin/env python3
"""
Real-time object detection with YOLO and RTSP streaming for Jetson
- Uses jetson_utils for hardware-accelerated camera capture and RTSP streaming
- Integrates Ultralytics YOLO model for object detection
- Draws bounding boxes and labels on detected objects
- Displays FPS and inference time metrics
"""

import time
import sys
import threading
from jetson_utils import videoSource, videoOutput, cudaFont, cudaDrawRect, cudaToNumpy
from rtsp_server import create_rtsp_server
from ultralytics import YOLO

def main():
    # Configuration
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    CAMERA_SOURCE = 'csi://0'
    OUTPUT_STREAM = 'rtp://0.0.0.0:1234'
    FLIP_METHOD = 'rotate-180'
    CONFIDENCE_THRESHOLD = 0.4
    MODEL_PATH = 'best.engine'
    
    print("Starting object detection with RTSP streaming...")
    
    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    try:
        input_camera = videoSource(CAMERA_SOURCE, options={
            'width': INPUT_WIDTH, 
            'height': INPUT_HEIGHT, 
            'flipMethod': FLIP_METHOD
        })
        print(f"Camera initialized: {INPUT_WIDTH}x{INPUT_HEIGHT}")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    
    # Initialize output stream
    try:
        output_stream = videoOutput(OUTPUT_STREAM)
        print(f"Output stream initialized: {OUTPUT_STREAM}")
    except Exception as e:
        print(f"Error initializing output stream: {e}")
        return
    
    # Start RTSP server in a separate thread
    try:
        threading.Thread(target=create_rtsp_server, daemon=True).start()
        print("RTSP server started at rtsp://0.0.0.0:8554/cam")
    except Exception as e:
        print(f"Error starting RTSP server: {e}")
    
    # Initialize CUDA font for text overlay
    font = cudaFont()
    
    # Define class names based on your dataset
    class_names = [
        'back_glass', 'back_left_side_light', 'back_right_side_light', 'door', 
        'front_bumper', 'front_glass', 'front_left_side_door', 'front_left_side_light', 
        'front_left_side_mirror', 'front_right_side_door', 'front_right_side_light', 
        'front_right_side_mirror', 'hood', 'rear_bumper', 'wheel'
    ]
    
    # Define colors for different classes (BGR format)
    colors = {
        0: (0, 255, 0),      # Green - back_glass
        1: (0, 0, 255),      # Red - back_left_side_light
        2: (255, 0, 0),      # Blue - back_right_side_light
        3: (255, 255, 0),    # Cyan - door
        4: (0, 255, 255),    # Yellow - front_bumper
        5: (255, 0, 255),    # Magenta - front_glass
        6: (128, 0, 0),      # Dark blue - front_left_side_door
        7: (0, 128, 0),      # Dark green - front_left_side_light
        8: (0, 0, 128),      # Dark red - front_left_side_mirror
        9: (128, 128, 0),    # Teal - front_right_side_door
        10: (0, 128, 128),   # Brown - front_right_side_light
        11: (128, 0, 128),   # Purple - front_right_side_mirror
        12: (64, 128, 255),  # Orange - hood
        13: (255, 128, 64),  # Light blue - rear_bumper
        14: (255, 255, 255), # White - wheel
    }
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    try:
        print("Starting detection and streaming loop...")
        # Main capture, detection, and streaming loop
        while True:
            # Capture frame with timeout
            try:
                cuda_img = input_camera.Capture(format='rgb8', timeout=5000)
                if cuda_img is None:
                    print("Camera capture timeout, retrying...")
                    time.sleep(0.1)  # Short delay before retry
                    continue
            except Exception as capture_error:
                print(f"Error capturing frame: {capture_error}")
                time.sleep(0.5)  # Longer delay on error
                continue
            
            # Start timing for inference
            inference_start = time.time()
            
            # Convert CUDA image to numpy array for YOLO processing
            try:
                numpy_img = cudaToNumpy(cuda_img)
                
                # Run detection
                results = model(numpy_img, conf=CONFIDENCE_THRESHOLD)
                
                # Calculate inference time
                inference_time = (time.time() - inference_start) * 1000  # ms
                
                # Debug: Print detection count
                detection_count = 0
                
                # Process detection results
                for result in results:
                    boxes = result.boxes
                    detection_count = len(boxes)
                    print(f"Detected {detection_count} objects")
                    
                    for box in boxes:
                        try:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Get class ID and confidence
                            class_id = int(box.cls[0].item())
                            confidence = box.conf[0].item()
                            
                            # Debug: Print detection details
                            print(f"  Class {class_id} ({class_names[class_id] if class_id < len(class_names) else 'unknown'}): "
                                  f"conf={confidence:.2f}, coords=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                            
                            # Get color for this class (default to white if not in colors dict)
                            color = colors.get(class_id, (255, 255, 255))
                            
                            # Get class name (use our predefined names)
                            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                            
                            # Create label with class name and confidence
                            label = f"{class_name} {confidence:.2f}"
                            
                            # Draw rectangle with explicit int conversion
                            cudaDrawRect(cuda_img, (int(x1), int(y1), int(x2), int(y2)), color, 2)
                            
                            # Draw text label
                            font.OverlayText(cuda_img, text=label, x=int(x1), y=int(y1-20), color=color)
                        except Exception as box_error:
                            print(f"Error processing box: {box_error}")
                
                # If no detections, draw a debug message on screen
                if detection_count == 0:
                    font.OverlayText(cuda_img, text="No detections", x=10, y=90, color=(0, 0, 255))
            except Exception as detection_error:
                print(f"Error in detection: {detection_error}")
                inference_time = 0
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"FPS: {fps:.1f}, Inference: {inference_time:.1f}ms")
            
            # Display FPS and inference time on the frame
            font.OverlayText(cuda_img, text=f"FPS: {fps:.1f}", x=10, y=30, color=(0, 255, 0))
            font.OverlayText(cuda_img, text=f"Inference: {inference_time:.1f}ms", x=10, y=60, color=(0, 255, 0))
            
            # Render the output frame
            try:
                output_stream.Render(cuda_img)
            except Exception as render_error:
                print(f"Error rendering frame: {render_error}")
            
            # Exit if streaming stops
            if not input_camera.IsStreaming() or not output_stream.IsStreaming():
                print("Streaming stopped")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        print("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
