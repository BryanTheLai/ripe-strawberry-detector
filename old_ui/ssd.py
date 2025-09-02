import gradio as gr
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# --- Import Model Definition ---
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.8
NUM_CLASSES = 2  # 1 class (strawberry) + 1 background

# --- Load the Trained Model ---
CKPT_PATH = os.path.join("models_for_streamlit", "best_ssd300_vgg16_model.pth")
print(f"Loading the trained SSD model from: {CKPT_PATH}")
model = ssd300_vgg16()
in_channels = [512, 1024, 512, 256, 256, 256]
num_anchors = model.anchor_generator.num_anchors_per_location()
new_head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=NUM_CLASSES)
model.head = new_head
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"SSD checkpoint not found: {CKPT_PATH}\nPlease place the checkpoint in the models_for_streamlit folder.")
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

# --- Helper Functions ---
def is_red_present(image_crop, red_threshold=0.1):
    """Checks if a significant portion of an image crop is red."""
    hsv_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_crop, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_crop, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    total_pixels = image_crop.shape[0] * image_crop.shape[1]
    if total_pixels == 0: return False
    red_pixel_count = cv2.countNonZero(red_mask)
    return (red_pixel_count / total_pixels) > red_threshold

def suppress_containing_boxes(boxes, scores):
    """
    Suppresses larger boxes that contain smaller boxes.
    If box 'A' contains box 'B', box 'A' is removed.
    """
    if len(boxes) == 0:
        return boxes, scores

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    indices_to_discard = set()

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j: continue
            # Check if box i (potentially larger) contains box j (potentially smaller)
            if (x1[i] < x1[j]) and (y1[i] < y1[j]) and (x2[i] > x2[j]) and (y2[i] > y2[j]):
                indices_to_discard.add(i)

    keep_mask = np.array([i not in indices_to_discard for i in range(len(boxes))])
    return boxes[keep_mask], scores[keep_mask]

# --- Main Detection Function ---
def detect_strawberries(input_image):
    """Takes an image, runs detection, filters results, and draws boxes."""
    inference_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = inference_transform(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    high_conf_mask = predictions['scores'] > CONFIDENCE_THRESHOLD
    boxes_after_conf = predictions['boxes'][high_conf_mask]
    scores_after_conf = predictions['scores'][high_conf_mask]
    
    nms_indices = torchvision.ops.nms(boxes_after_conf, scores_after_conf, iou_threshold=0.45)
    boxes_after_nms = boxes_after_conf[nms_indices]
    scores_after_nms = scores_after_conf[nms_indices]
    
    output_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    h, w, _ = output_image.shape
    scale_x, scale_y = w / 300, h / 300

    # STEP 1: Pre-filter to find all boxes containing red
    valid_red_boxes, valid_red_scores = [], []
    for i in range(len(boxes_after_nms)):
        box = boxes_after_nms[i].cpu().numpy()
        x1, y1 = int(box[0] * scale_x), int(box[1] * scale_y)
        x2, y2 = int(box[2] * scale_x), int(box[3] * scale_y)
        crop_x1, crop_y1 = max(0, x1), max(0, y1)
        crop_x2, crop_y2 = min(w, x2), min(h, y2)

        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            detected_crop = output_image[crop_y1:crop_y2, crop_x1:crop_x2]
            if is_red_present(detected_crop):
                valid_red_boxes.append(boxes_after_nms[i])
                valid_red_scores.append(scores_after_nms[i])
    
    print(f"Boxes containing red: {len(valid_red_boxes)}")

    # STEP 2: From valid boxes, suppress any that contain a smaller one
    if len(valid_red_boxes) > 0:
        valid_red_boxes_tensor = torch.stack(valid_red_boxes)
        valid_red_scores_tensor = torch.stack(valid_red_scores)
        
        final_boxes, final_scores = suppress_containing_boxes(
            valid_red_boxes_tensor.cpu().numpy(), 
            valid_red_scores_tensor.cpu().numpy()
        )
        print(f"Boxes after suppression: {len(final_boxes)}")
    else:
        final_boxes, final_scores = [], []
    
    # STEP 3: Draw only the final, filtered boxes
    for i in range(len(final_scores)):
        score, box = final_scores[i], final_boxes[i]
        x1, y1 = int(box[0] * scale_x), int(box[1] * scale_y)
        x2, y2 = int(box[2] * scale_x), int(box[3] * scale_y)
        
        label = f"Strawberry: {score:.2f}"
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# --- Create and Launch the Gradio Interface ---
iface = gr.Interface(
    fn=detect_strawberries,
    inputs=gr.Image(type="pil", label="Upload Strawberry Image", sources=["upload"]),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="🍓 Strawberry Detection with SSD300_VGG16",
    description="Upload an image to detect strawberries. This interface suppresses large boxes that contain smaller detections."
)

if __name__ == "__main__":
    iface.launch()