from ultralytics import YOLO

# Load a pretrained YOLO11n model
#model = YOLO("/mnt/c/detection test/runs/detect/train5/weights/best.pt")
model = YOLO("/mnt/c/detection test/runs/detect/train10/weights/best.pt")
# Run inference on 'bus.jpg' with arguments
model.predict("/mnt/c/detection test/images.jpg", save=True, conf=0.7)
#model.predict("/mnt/c/detection test/car.mp4", save=True, conf=0.8)