#!/usr/bin/env python3
"""
Basic camera streaming test script for Jetson platform
- Uses jetson_utils for hardware-accelerated camera capture and RTSP streaming
- Focuses on establishing a stable video pipeline before adding detection
"""

import time
import sys
import threading
from jetson_utils import videoSource, videoOutput, cudaFont, cudaDeviceSynchronize
from rtsp_server import create_rtsp_server

# Configuration
INPUT_WIDTH = 1280
INPUT_HEIGHT = 720
CAMERA_SOURCE = 'csi://0'
OUTPUT_STREAM = 'rtp://0.0.0.0:1234'
FLIP_METHOD = 'rotate-180'  # Options: none, counterclockwise, rotate-180, clockwise, horizontal, vertical

def main():
    print("Starting basic camera streaming test...")
    
    # Initialize camera with longer timeout
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
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    try:
        print("Starting capture loop...")
        # Main capture and streaming loop
        while True:
            loop_start = time.time()
            
            # Capture frame with timeout
            try:
                cuda_img = input_camera.Capture(format='rgb8', timeout=5000)  # Increased timeout
                if cuda_img is None:
                    print("Camera capture timeout, retrying...")
                    time.sleep(0.1)  # Short delay before retry
                    continue
            except Exception as capture_error:
                print(f"Error capturing frame: {capture_error}")
                time.sleep(0.5)  # Longer delay on error
                continue
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            # Display FPS on the frame
            font.OverlayText(cuda_img, text=f"FPS: {fps:.1f}", x=10, y=30, color=(0, 255, 0))
            
            # Ensure CUDA operations are complete before rendering
            cudaDeviceSynchronize()
            
            # Render the output frame
            try:
                output_stream.Render(cuda_img)
            except Exception as render_error:
                print(f"Error rendering frame: {render_error}")
            
            # Exit if streaming stops
            if not input_camera.IsStreaming() or not output_stream.IsStreaming():
                print("Streaming stopped")
                break
            
            # Calculate loop time for debugging
            loop_time = (time.time() - loop_start) * 1000
            if loop_time > 100:  # Log if processing takes more than 100ms
                print(f"Long frame processing time: {loop_time:.1f}ms")
                
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
