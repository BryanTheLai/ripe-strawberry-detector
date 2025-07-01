"""
train_and_infer.py

Provides functions to train YOLO models and run inference for strawberry detection.
"""

from ultralytics import YOLO
import os
import logging
from src.baseline import create_template, run_baseline
from src.yolo_utils import split_and_prepare
from typing import Union

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def train_model(data_yaml: str,
                model_name: str = 'yolo11s.pt',
                epochs: int = 100,
                imgsz: int = 640,
                device: Union[str, int] = 0,
                patience: int = 50) -> str:
    """Train a YOLOv11 model on the specified dataset.

    Args:
        data_yaml: Path to data config YAML.
        model_name: YOLO model weights or name.
        epochs: Number of training epochs.
        imgsz: Image size for training.
        device: Compute device ('cpu', 'cuda', or index).
        patience: Early stopping patience in epochs (stop if no improvement).

    Returns:
        Path to the best saved weights file.
    """
    # Train YOLOv11 model and save weights under output/train/weights/best.pt
    yolo_model = YOLO(model_name)
    yolo_model.train(data=data_yaml,
                     epochs=epochs,
                     imgsz=imgsz,
                     device=device,
                     patience=patience,
                     project='output',
                     name='train',
                     exist_ok=True)
    weights_path = os.path.join('output', 'train', 'weights', 'best.pt')
    logging.info(f"Training complete. Best weights at: {weights_path}")
    return weights_path

def run_inference(model_path: str,
                  images_dir: str,
                  output_dir: str,
                  conf: float = 0.25) -> int:
    """Run inference using a YOLO model and count strawberry detections.

    Args:
        model_path: Path to trained YOLO weights.
        images_dir: Directory or image file for inference.
        output_dir: Directory to save prediction visuals.
        conf: Confidence threshold for predictions.

    Returns:
        Number of strawberry detections.
    """
    # Run inference on a directory of images and return count of strawberry detections
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    # predict with saving results under output_dir/predictions and capture results
    results = model.predict(source=images_dir,
                              conf=conf,
                              save=True,
                              project=output_dir,
                              name='predictions',
                              exist_ok=True)
    # Determine 'strawberry' class ID
    straw_id = next((cid for cid, name in model.names.items() if name.lower() == 'strawberry'), None)
    strawberry_count = 0
    for r in results:
        cls_ids = getattr(r.boxes, 'cls', None)
        if cls_ids is not None and straw_id is not None:
            strawberry_count += int((cls_ids == straw_id).sum().item())
    logging.info(f"Inference complete. Found {strawberry_count} strawberry detections.")
    return strawberry_count

def main():
    # Paths
    cwd = os.getcwd()
    xml_path = os.path.join(cwd, 'dataset', 'annotations.xml')
    images_dir = os.path.join(cwd, 'dataset', 'images')
    template_path = os.path.join(cwd, 'template', 'template.png')
    baseline_out = os.path.join(cwd, 'output', 'baseline_result.png')
    # Run baseline
    logging.info('Creating template...')
    create_template(xml_path, images_dir, template_path)
    logging.info('Running OpenCV baseline...')
    run_baseline(os.path.join(images_dir, '0.png'), template_path, baseline_out)

    # Prepare YOLO dataset
    yolo_dataset_dir = os.path.join(cwd, 'strawberry_dataset')
    logging.info('Preparing YOLO dataset...')
    split_and_prepare(xml_path, images_dir, yolo_dataset_dir)

    # Train YOLO model
    data_yaml = os.path.join(yolo_dataset_dir, 'data.yaml')
    logging.info('Training YOLOv11 model...')
    weights_path = train_model(data_yaml, device=0)

    # Run inference on validation set
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    yolo_out = os.path.join(cwd, 'output', 'yolo_results')
    logging.info('Running YOLO inference on validation images...')
    run_inference(weights_path, val_dir, yolo_out)

if __name__ == '__main__':
    main()
