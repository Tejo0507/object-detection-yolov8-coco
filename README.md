# Object Detection Project using YOLOv8

## Dataset

- Used a subset of the COCO dataset prepared via Roboflow.
- Dataset structure follows YOLO format with folders:
  - `images/train` and `images/val` containing training and validation images.
  - `labels/train` and `labels/val` containing corresponding annotation files in YOLO `.txt` format.
- The dataset config file (`data.yaml`) specifies paths and class names.

## Preprocessing

- No custom preprocessing applied beyond Roboflow’s standardized formatting.
- YOLOv8 automatically resizes and normalizes images during training.
- Validation and training images were verified for consistency.

## Workflow

### Training

- Used `train.py` to train YOLOv8 model with the COCO subset.
- Training checkpoint weights saved in `runs/train/exp/weights/` with `best.pt` and `last.pt`.

### Inference

- Used `infer.py` to run predictions on validation images.
- Output images with bounding boxes can be saved.

### Visualization

- Used `visualize.py` to display detected bounding boxes and class labels interactively.

### Evaluation

- Used `evaluate.py` to assess model performance metrics including precision, recall, and mAP.

## Results

- Achieved mAP @[IoU=0.5] of **~60% to 77%** on top-performing classes.
- Model detects classes such as **zebra, train, toaster, teddy bear, sheep, and toilet** with strong accuracy.
- Clear bounding box visualizations with confidence scores were observed during inference.

### Example per-class mAP@0.5 (rounded):

| Class       | mAP@0.5 |
|-------------|----------|
| Zebra       | 83.1%    |
| Train       | 75.8%    |
| Toaster     | 34.6%    |
| Teddy Bear  | 55.6%    |
| Sheep       | 62.9%    |
| Toilet      | 66.2%    |

- Inference speed is approximately **1.5ms per image**, showing efficient detection performance.

## How to Run

1. Install dependencies from `requirements.txt`.
2. Train the model:   
python scripts/train.py
3. Run inference:   
python scripts/infer.py
4. Visualize detections:   
python scripts/visualize.py
5. Evaluate model:   
python scripts/evaluate.py
