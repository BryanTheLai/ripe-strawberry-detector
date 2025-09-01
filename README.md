# Ripe Strawberry Detector

Detect and count ripe strawberries in images using two approaches:
- **Template Matching (OpenCV)**
- **Fine-tuned YOLOv11 (Deep Learning)**

## Quickstart

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/ripe-strawberry-detector.git
   cd ripe-strawberry-detector
   ```

2. **Set up the environment**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - [Strawberries Dataset (Kaggle)](https://www.kaggle.com/datasets/trainingdatapro/ripe-strawberries-detection)
   - Place images and annotations in the dataset folder as described below.

4. **Run the notebook**
   - Open `yolo11s_demo.ipynb` in VS Code or Jupyter.
   - Run all cells to see the pipeline: data prep, baseline, YOLO training, and evaluation.

5. **Run the Streamlit App**
   - Launch the interactive web UI for strawberry detection:
     ```sh
     streamlit run streamlit_ui.py
     ```
   - Upload an image and select the model (pretrained or finetuned YOLO) to detect and count strawberries with bounding boxes.
   - Note: the generic pretrained model does not include a "strawberry" class; use your finetuned weights for counting.

---

## Project Structure

```
dataset/
  images/           # Raw images
  annotations.xml   # Original annotation file
output/
  ...               # Results, predictions, and plots
src/
  baseline.py       # Template matching logic
  train_and_infer.py    # YOLO training/inference
  yolo_utils.py     # Data prep utilities
strawberry_dataset/ # YOLO-formatted dataset (auto-generated)
template/
  template.png      # Template image for baseline
requirements.txt
yolo11s_demo.ipynb
```

---

## Algorithms

### 1. Baseline: Template Matching (OpenCV)
- Fast, no training required.
- Sensitive to scale, rotation, and lighting.
- Usage: Crops a template from the dataset and matches it across images.

### 2. Fine-tuned YOLOv11 (Deep Learning)
- Trains a YOLOv11 model on your labeled strawberry dataset.
- Robust to scale, rotation, and lighting.
- Requires labeled data and training time.

---

## How to Use

1. **Prepare the Dataset**
   - Place all images in images.
   - Place the annotation XML in annotations.xml.

2. **Run Data Preparation**
   - The notebook will split and convert the dataset to YOLO format automatically.

3. **Run All Algorithms**
   - The notebook runs the baseline and YOLO on a sample image and compares results.
   - Training YOLO may take time (GPU recommended).

4. **Results**
   - Output images and detection counts are saved in the output directory.
   - Visual and numerical comparisons are shown in the notebook.

---

## Customization

- **Change the sample image:** Edit the `SAMPLE_IMAGE_NAME` variable in the notebook.
- **Train longer:** Increase the `epochs` parameter in the YOLO training cell for better accuracy.

---

## Requirements

- Python 3.10+
- See requirements.txt for all dependencies.

---

## Troubleshooting

- **No detections?** Check your dataset paths and annotation format.
- **YOLO training slow?** Use a machine with a CUDA-compatible GPU.
- **Pretrained model counts 0?** It likely does not include a "strawberry" class. Use finetuned weights.
- **Video UI?** Out of scope for this project version.

---

## License

MIT License.

---

## Acknowledgments

- [Kaggle: Ripe Strawberries Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/ripe-strawberries-detection)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

This README is ready for future updates, including count extraction and additional algorithms. Let me know when you want to add those features.