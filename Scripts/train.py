from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    model.train(
        data='E:/ObjectDetection/Dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=device
    )

if __name__ == '__main__':
    main()
