# DrainMind — Model Weights

## Overview

DrainMind uses the Ultralytics YOLOv8 framework for object detection.
By default, the application downloads `yolov8n.pt` (≈6 MB) automatically on first inference
if network access is available. This pretrained model was trained on the COCO dataset.

---

## Important Limitation

**A COCO-pretrained model does NOT specifically recognise drainage blockage, standing water,
leaf debris, or drain grids.** It recognises 80 general COCO object classes.

Results obtained without custom drainage-specific weights are **exploratory only**.
Coverage estimates and inspection priority scores should be interpreted accordingly.

---

## Where to Place Custom Weights

Place your `.pt` file in this directory, then configure the path in the application:

1. Open the application and navigate to **Settings**.
2. Enter the absolute path to your `.pt` file in the **Custom model weights path** field.
3. Click **Save Settings**.
4. The detector will reload on the next analysis.

Example:
```
models/drain_yolov8n_custom.pt
```

---

## Supported Model Formats

- **YOLOv8 PyTorch** (`.pt`) — recommended
- **YOLOv8 ONNX** (`.onnx`) — supported by Ultralytics
- **Custom trained YOLO** — any YOLOv8 variant with drainage-specific classes

---

## Training a Drainage-Specific Model

### Recommended class labels

```
0: drain_blocked
1: drain_partial
2: drain_clear
3: standing_water
4: debris
5: leaves
6: trash
```

Adjust classes to match your inspection requirements.

### Dataset preparation

1. Collect images of drainage infrastructure from multiple locations and lighting conditions.
2. Annotate using LabelImg, Roboflow, or CVAT in YOLO format.
3. **Split by location or recording session** — NOT randomly across frames from the same video.
   Random splitting of video frames creates data leakage (near-identical frames in train and val).
4. Recommended split: 70% train / 15% val / 15% test.
5. Each video should contribute frames to only one split.

### Training command (example)

```bash
yolo detect train \
  data=configs/drain_dataset.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=cpu \
  project=runs/drain_detect \
  name=v1
```

### Dataset YAML template

```yaml
path: /path/to/DrainMind/data/train_dataset
train: images/train
val: images/val
test: images/test

nc: 7
names:
  - drain_blocked
  - drain_partial
  - drain_clear
  - standing_water
  - debris
  - leaves
  - trash
```

---

## Running Inference Offline

Once you have a `.pt` file, the application operates completely offline.
No external API calls are made during inference.

Set the model path in Settings and the application will load the weights locally.

---

## License Considerations

- `yolov8n.pt` (COCO pretrained) — Ultralytics AGPL-3.0 License
  ([https://github.com/ultralytics/ultralytics/blob/main/LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE))
- Custom-trained weights inherit the base model's license unless you train from scratch.
- For commercial use, contact Ultralytics for an enterprise licence.
- DrainMind is intended for academic, research, and non-commercial demonstration use.

---

## Verifying a Model

```python
from ultralytics import YOLO
model = YOLO("models/your_model.pt")
print(model.names)  # Should list drainage-specific class names
results = model.predict("path/to/test_image.jpg", conf=0.35)
results[0].show()
```
