from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/ObjectDetection/runs/detect/train/weights/last.pt')
    results = model.val(data='E:/ObjectDetection/Dataset/data.yaml', imgsz=640)
    print(results.metrics)
