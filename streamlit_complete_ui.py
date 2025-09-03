import os
import io
import tempfile
import time

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

import torch
import torchvision.transforms as T
import torchvision

# Try importing YOLO (ultralytics); if missing we'll show an informative error in the UI
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

st.set_page_config(page_title="Complete Strawberry Detector", layout="wide")
st.title("🍓 Complete Strawberry Detector — RetinaNet / SSD / YOLO")

# ----------------------------
# Sidebar: device, model choice, shared settings
# ----------------------------
st.sidebar.header("Settings")

# Model selection
MODEL_GROUP = st.sidebar.selectbox("Model family", ["RetinaNet", "SSD300_VGG16", "YOLO (Ultralytics)"])

# Model paths (point to models_for_streamlit by default)
RETINA_CKPT = os.path.join("models_for_streamlit", "retinanet_best.pt")
SSD_CKPT = os.path.join("models_for_streamlit", "best_ssd300_vgg16_model.pth")
YOLO_CKPT = os.path.join("models_for_streamlit", "yolo11s.pt")
YOLO_CKPT_FT = os.path.join("models_for_streamlit", "yolo_ft_best.pt")

# Configure sidebar per selected model family
HAS_CUDA = torch.cuda.is_available()

if MODEL_GROUP == "SSD300_VGG16":
    # SSD uses fixed checkpoint and defaults from ssd.py; hide interactive settings
    st.sidebar.caption("SSD uses a fixed checkpoint and internal settings. No sidebar options.")
    ckpt_path = SSD_CKPT
    SSD_CONF_DEFAULT = 0.8
    # Device: follow ssd.py behavior (auto cuda if available)
    DEVICE = torch.device('cuda' if HAS_CUDA else 'cpu')
    device_choice = "cuda" if HAS_CUDA else "cpu"
    # set placeholders for variables used later
    show_scores = True
    thr = SSD_CONF_DEFAULT
    nms = 0.34
    max_det = 50

elif MODEL_GROUP == "RetinaNet":
    # RetinaNet exposes presets and sliders (see retina_streamlit.py)
    device_choice = st.sidebar.selectbox("Device", ["CPU"] + (["CUDA"] if HAS_CUDA else []))
    DEVICE = torch.device("cuda:0" if (device_choice == "CUDA" and HAS_CUDA) else "cpu")

    default_ckpt = RETINA_CKPT
    ckpt_path = st.sidebar.text_input("RetinaNet checkpoint (.pt)", value=default_ckpt)

    if "thr" not in st.session_state: st.session_state.thr = 0.42
    if "nms" not in st.session_state: st.session_state.nms = 0.34
    if "max_det" not in st.session_state: st.session_state.max_det = 50

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Metric best\n(0.34 / 0.34)"):
        st.session_state.thr = 0.34
        st.session_state.nms = 0.34
        st.session_state.max_det = 300
    if c2.button("Visual clean\n(0.42 / 0.34)"):
        st.session_state.thr = 0.42
        st.session_state.nms = 0.34
        st.session_state.max_det = 50

    thr = st.sidebar.slider("Confidence threshold", 0.00, 1.00, st.session_state.thr, 0.01, key="thr")
    nms = st.sidebar.slider("NMS IoU", 0.30, 0.70, st.session_state.nms, 0.01, key="nms")
    max_det = st.sidebar.slider("Max detections/image", 10, 300, st.session_state.max_det, 10, key="max_det")
    show_scores = st.sidebar.checkbox("Show scores on boxes", value=True)

elif MODEL_GROUP == "YOLO (Ultralytics)":
    # YOLO uses its own device selector and confidence slider (see streamlit_ui_yolo.py)
    device_options = ["cpu"]
    if HAS_CUDA:
        device_options.append("cuda:0")
    device_choice = st.sidebar.selectbox("Compute device", device_options)
    # keep DEVICE as torch.device where needed; YOLO.predict accepts device string as well
    DEVICE = torch.device("cuda" if device_choice.startswith("cuda") else "cpu")

    yolo_variant = st.sidebar.selectbox("YOLO variant", ["Pretrained YOLO (yolo11s.pt)", "Finetuned YOLO (yolo_ft_best.pt)"])
    default_yolo = YOLO_CKPT_FT if yolo_variant.startswith("Finetuned") else YOLO_CKPT
    ckpt_path = st.sidebar.text_input("YOLO weights (.pt)", value=default_yolo)

    thr = st.sidebar.slider("Confidence threshold", 0.00, 1.00, 0.5, 0.01)
    # For YOLO we use 'nms' and 'max_det' defaults but they are not critical here
    nms = 0.34
    max_det = 50
    show_scores = True

else:
    # fallback defaults
    DEVICE = torch.device('cpu')
    device_choice = 'cpu'
    thr = 0.42
    nms = 0.34
    max_det = 50
    show_scores = True

# small layout polish
st.sidebar.markdown("---")
st.sidebar.caption("Choose a model family to reveal its settings. SSD uses its internal defaults.")

# ----------------------------
# RetinaNet loader + helpers (reused from retina_streamlit.py)
# ----------------------------
to_tensor = T.ToTensor()
CLASS_ID = 1

@st.cache_resource(show_spinner=True)
def load_retinanet(ckpt_path: str, device: torch.device):
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    model.eval()

    anchor_sizes = (
        (16, 32, 64),
        (32, 64, 128),
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
    )
    aspect_ratios = (0.5, 1.0, 2.0)
    model.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=[aspect_ratios] * len(anchor_sizes),
    )

    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes=2)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nPlease place the checkpoint in the models_for_streamlit folder.")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model

def run_retinanet(model, pil_img, score_thresh=0.42, nms_thresh=0.34, max_dets=50):
    if hasattr(model, "score_thresh"):        model.score_thresh = float(score_thresh)
    if hasattr(model, "nms_thresh"):          model.nms_thresh = float(nms_thresh)
    if hasattr(model, "detections_per_img"):  model.detections_per_img = int(max_dets)

    img_t = to_tensor(pil_img).to(DEVICE)
    model.eval()
    with torch.inference_mode():
        out = model([img_t])[0]

    boxes  = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy().astype(int)
    take = (labels == CLASS_ID)
    return boxes[take], scores[take]

# reuse PIL drawing from retina_streamlit
def draw_boxes_pil(pil_img, boxes, scores=None, show_scores=True):
    img = pil_img.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i].tolist()]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        if show_scores and scores is not None:
            s = float(scores[i]); txt = f"{s:.2f}"
            tw = d.textlength(txt, font=font)
            th = 16
            d.rectangle([x1, max(0, y1 - th - 2), x1 + tw + 6, y1], fill=(255, 0, 0))
            d.text((x1 + 3, max(0, y1 - th - 2)), txt, fill=(255, 255, 255), font=font)
    return img

# ----------------------------
# SSD loader + helper (reused from ssd.py)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_ssd(ckpt_path: str, device: torch.device):
    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSDHead

    model = ssd300_vgg16()
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    new_head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=2)
    model.head = new_head

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SSD checkpoint not found: {ckpt_path}\nPlease place the checkpoint in the models_for_streamlit folder.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def is_red_present(image_crop, red_threshold=0.1):
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
    if len(boxes) == 0:
        return boxes, scores

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    indices_to_discard = set()

    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j: continue
            if (x1[i] < x1[j]) and (y1[i] < y1[j]) and (x2[i] > x2[j]) and (y2[i] > y2[j]):
                indices_to_discard.add(i)

    keep_mask = np.array([i not in indices_to_discard for i in range(len(boxes))])
    return boxes[keep_mask], scores[keep_mask]

def run_ssd_detection(model, pil_img, conf_thresh=0.8):
    # reuse transforms from original ssd.py
    inference_transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = inference_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    high_conf_mask = predictions['scores'] > conf_thresh
    boxes_after_conf = predictions['boxes'][high_conf_mask]
    scores_after_conf = predictions['scores'][high_conf_mask]

    if boxes_after_conf.shape[0] == 0:
        return np.zeros((0,4)), np.zeros((0,))

    nms_indices = torchvision.ops.nms(boxes_after_conf, scores_after_conf, iou_threshold=nms)
    boxes_after_nms = boxes_after_conf[nms_indices]
    scores_after_nms = scores_after_conf[nms_indices]

    # Convert boxes (which are in 0..300 input coords) to original image pixel coords
    output_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = output_image.shape
    scale_x, scale_y = w / 300.0, h / 300.0

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
                # keep original (0..300) coords for suppression
                valid_red_boxes.append(boxes_after_nms[i].cpu().numpy())
                valid_red_scores.append(scores_after_nms[i].cpu().numpy())

    if len(valid_red_boxes) == 0:
        return np.zeros((0,4)), np.zeros((0,))

    valid_red_boxes = np.stack(valid_red_boxes)
    valid_red_scores = np.stack(valid_red_scores)
    final_boxes, final_scores = suppress_containing_boxes(valid_red_boxes, valid_red_scores)

    # scale final_boxes to pixel coords
    pixel_boxes = []
    for box in final_boxes:
        x1 = int(box[0] * scale_x); y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x); y2 = int(box[3] * scale_y)
        pixel_boxes.append([x1, y1, x2, y2])
    if len(pixel_boxes) == 0:
        return np.zeros((0,4)), np.zeros((0,))
    return np.array(pixel_boxes), np.array(final_scores)

# ----------------------------
# YOLO helpers (reused from streamlit_ui_yolo.py)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_yolo(ckpt_path: str):
    if not HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics package not available in this environment. Install ultralytics to use YOLO models.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"YOLO weights not found at: {ckpt_path}")
    model = YOLO(ckpt_path)
    return model

# ----------------------------
# Main UI flow
# ----------------------------
uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"]) 

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")

    # Show preview (make it larger / full width). Run Detection button sits under the preview area
    st.markdown("---")
    st.image(pil, use_container_width=True)

    if st.button("Run Detection"):
        t0 = time.time()
        try:
            if MODEL_GROUP == "RetinaNet":
                model = load_retinanet(ckpt_path, DEVICE)
                boxes, scores = run_retinanet(model, pil, score_thresh=thr, nms_thresh=nms, max_dets=max_det)
                vis = draw_boxes_pil(pil, boxes, scores, show_scores=show_scores)

            elif MODEL_GROUP == "SSD300_VGG16":
                model = load_ssd(ckpt_path, DEVICE)
                # Use SSD's original confidence default (0.8) rather than the global slider
                boxes, scores = run_ssd_detection(model, pil, conf_thresh=SSD_CONF_DEFAULT)
                vis = draw_boxes_pil(pil, boxes, scores, show_scores=show_scores)

            else:  # YOLO
                model = load_yolo(ckpt_path)
                # write to temp file and call model.predict like in original
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                pil.save(tmp_file.name)
                tmp_file.close()
                res = model.predict(source=tmp_file.name, conf=thr, save=False, device=device_choice)
                os.remove(tmp_file.name)
                # extract boxes
                if len(res) == 0 or len(res[0].boxes) == 0:
                    boxes = np.zeros((0,4))
                    scores = np.zeros((0,))
                else:
                    boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
                    scores = res[0].boxes.conf.cpu().numpy()
                vis = draw_boxes_pil(pil, boxes, scores, show_scores=show_scores)

            dt = time.time() - t0
            count = 0 if boxes.size == 0 else len(boxes)

            # Show detection stats in a more prominent way
            st.markdown("### 🔍 Detection Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detections Found", count)
            with col2:
                st.metric("Runtime", f"{dt:.2f}s")
            with col3:
                st.metric("Device", DEVICE.type.upper())
            
            # Additional parameters in an info box
            st.info(f"**Parameters:** Confidence threshold: {thr:.2f} • NMS IoU: {nms:.2f} • Max detections: {max_det}")

            # Only show the detection image (take full width). We no longer display the original alongside it.
            st.markdown("### Detection")
            st.image(vis, use_container_width=True)

            # Download annotated image
            buf = io.BytesIO()
            vis.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="Download annotated image (PNG)",
                data=buf.getvalue(),
                file_name=f"pred_{MODEL_GROUP}_thr{thr:.2f}.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(str(e))
else:
    st.info("Upload an image to begin. Use the sidebar to pick a model and device.")
