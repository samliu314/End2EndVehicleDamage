import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import time

custom_transforms = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(std_range=(0.05, 0.1), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load a model
model = YOLO("yolo11m.pt")  # build a new model from YAML

# Train the model
start_time = time.time()
results = model.train(data="/mnt/c/detection test/datasets/Test2/data.yaml", epochs=50, imgsz=640, lr0=0.0005, plots=True)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")
#results = model.train(data="/mnt/c/detection test/datasets/cartire.v1i.yolov11/data.yaml", epochs=30, imgsz=640, optimizer='AdamW', lr0=0.0004)