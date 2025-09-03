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

4. **Run the notebooks**

5. **Run the Streamlit App**
   - Launch the interactive web UI for strawberry detection:
     ```sh
     streamlit run streamlit_complete_ui.py
     ```

---

## Project Structure

```
dataset/
   images/             # Raw images
   boxes/              # Cropped strawberry boxes
   annotations.xml     # Original annotation file
   strawberries.csv    # CSV annotation (optional)
output/
   baseline_sample.jpg # Example output
   report_metrics.md   # Metrics report
   baseline_eval/      # Baseline predictions
   finetuned_preds/    # Finetuned YOLO predictions
   pretrained_preds/   # Pretrained YOLO predictions
   train/              # YOLO training outputs
   val/                # YOLO validation outputs
src/
   baseline.py         # Template matching logic
   train_and_infer.py  # YOLO training/inference
   yolo_utils.py       # Data prep utilities
models_for_streamlit/
   yolo_ft_best.pt     # Finetuned YOLO weights
   yolo11s.pt          # Pretrained YOLO weights
   retinanet_best.pt   # RetinaNet weights
   best_ssd300_vgg16_model.pth # SSD weights
strawberry_dataset/   # YOLO-formatted dataset (auto-generated)
template/
   template.png        # Template image for baseline
scripts/
   generate_report_metrics.py # Script for metrics
requirements.txt
streamlit_complete_ui.py     # Streamlit app
1_data_prep.ipynb            # Data prep notebook
2_bryan_part.ipynb           # Bryan's notebook
3_weilet_retina_net.ipynb    # RetinaNet notebook
4_tze_hong_ssd.ipynb         # SSD notebook
yolo11n.pt                   # YOLOv11n weights
yolo11s.pt                   # YOLOv11s weights
```

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

## Requirements

- Python 3.10+
- See requirements.txt for all dependencies.

---

## Troubleshooting

- **No detections?** Check your dataset paths and annotation format.
- **YOLO training slow?** Use a machine with a CUDA-compatible GPU.
- **Pretrained model counts 0?** It likely does not include a "strawberry" class. Use finetuned weights.

---

## License

MIT License.

---

## Acknowledgments

- [Kaggle: Ripe Strawberries Detection Dataset](https://www.kaggle.com/datasets/trainingdatapro/ripe-strawberries-detection)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)