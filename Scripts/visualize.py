from ultralytics import YOLO
import cv2

model = YOLO('E:/ObjectDetection/runs/detect/train/weights/last.pt')

results = model.predict(source='E:/ObjectDetection/Dataset/valid/images', conf=0.25, imgsz=640)

for result in results:
    img = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for (box, score, cls) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Detection', img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
