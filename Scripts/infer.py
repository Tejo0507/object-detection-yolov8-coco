from ultralytics import YOLO

model = YOLO('E:/ObjectDetection/runs/detect/train/weights/last.pt')

results = model.predict(source='E:/ObjectDetection/Dataset/valid/images', conf=0.25, imgsz=640, save=True)
print(results)
