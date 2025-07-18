from threading import Thread
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaFromNumpy
from rtsp_server import create_rtsp_server
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


THRESHOLD = 0.4
MODEL = "best.pt"
INPUT = "csi://0"
OUTPUT = "rtp://0.0.0.0:1234"
WIDTH = 1280
HEIGHT = 720


model = YOLO(MODEL)
input = videoSource(INPUT, options={'width': WIDTH, 'height': HEIGHT, 'flipMethod': 'rotate-180'})
# output = videoOutput("my_video.mp4", options={'codec': 'h264', 'bitrate': 4000000})
output = videoOutput(OUTPUT)

Thread(target=create_rtsp_server, daemon=True).start()

# capture frames until end-of-stream (or the user exits)
while True:
    image = input.Capture(format='rgb8', timeout=1000)  
	
    if image is None:  # if a timeout occurred
        continue

    frame = cudaToNumpy(image)
    results = model.predict(frame, conf=THRESHOLD, verbose=False)
    test = cudaFromNumpy(frame)
    output.Render(image)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break