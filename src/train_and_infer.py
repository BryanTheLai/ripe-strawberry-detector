from ultralytics import YOLO
import os
from src.baseline import create_template, run_baseline
from src.yolo_utils import split_and_prepare
from typing import Union

def train_model(data_yaml: str,
                model_name: str = 'yolo11s.pt',
                epochs: int = 100,
                imgsz: int = 640,
                device: Union[str, int] = 0):
    # Train YOLOv11 model and save weights under output/train/weights/best.pt
    yolo_model = YOLO(model_name)
    yolo_model.train(data=data_yaml,
                     epochs=epochs,
                     imgsz=imgsz,
                     device=device,
                     project='output',
                     name='train',
                     exist_ok=True)
    weights_path = os.path.join('output', 'train', 'weights', 'best.pt')
    return weights_path

def run_inference(model_path: str, images_dir: str, output_dir: str, conf: float = 0.25):
    # Run inference on a directory of images and save outputs
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    # predict with saving results under output_dir/predictions
    model.predict(source=images_dir,
                  conf=conf,
                  save=True,
                  project=output_dir, # Save to the specified output directory
                  name='predictions', # Save in a folder named 'predictions'
                  exist_ok=True) # Overwrite if the directory exists
    print(f"Inference complete. Results saved to {os.path.join(output_dir, 'predictions')}.")

def main():
    # Paths
    cwd = os.getcwd()
    xml_path = os.path.join(cwd, 'dataset', 'annotations.xml')
    images_dir = os.path.join(cwd, 'dataset', 'images')
    template_path = os.path.join(cwd, 'template', 'template.png')
    baseline_out = os.path.join(cwd, 'output', 'baseline_result.png')
    # Run baseline
    print('Creating template...')
    create_template(xml_path, images_dir, template_path)
    print('Running OpenCV baseline...')
    run_baseline(os.path.join(images_dir, '0.png'), template_path, baseline_out)

    # Prepare YOLO dataset
    yolo_dataset_dir = os.path.join(cwd, 'strawberry_dataset')
    print('Preparing YOLO dataset...')
    split_and_prepare(xml_path, images_dir, yolo_dataset_dir)

    # Train YOLO model
    data_yaml = os.path.join(yolo_dataset_dir, 'data.yaml')
    print('Training YOLOv11 model...')
    weights_path = train_model(data_yaml, device=0)

    # Run inference on validation set
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    yolo_out = os.path.join(cwd, 'output', 'yolo_results')
    print('Running YOLO inference on validation images...')
    run_inference(weights_path, val_dir, yolo_out)

if __name__ == '__main__':
    main()
