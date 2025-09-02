# retina_streamlit.py
import os
import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.anchor_utils import AnchorGenerator

import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="RetinaNet Strawberry Detector", layout="wide")
st.title("🍓 RetinaNet Strawberry Detector")
st.caption("Torchvision RetinaNet (ResNet-50 FPN v2) • medium anchors • 1 class + background")

# ----------------------------
# Sidebar: device, checkpoint, presets
# ----------------------------
st.sidebar.header("Settings")

HAS_CUDA = torch.cuda.is_available()
device_choice = st.sidebar.selectbox("Device", ["CPU"] + (["CUDA"] if HAS_CUDA else []))
DEVICE = torch.device("cuda:0" if (device_choice == "CUDA" and HAS_CUDA) else "cpu")

default_ckpt = os.path.join("models_for_streamlit", "retinanet_best.pt")
CKPT_PATH = st.sidebar.text_input("Checkpoint (.pt)", value=default_ckpt)

# --- Presets (store in session_state and allow quick switching)
if "thr" not in st.session_state: st.session_state.thr = 0.42  # visual clean default
if "nms" not in st.session_state: st.session_state.nms = 0.34
if "max_det" not in st.session_state: st.session_state.max_det = 50

c1, c2 = st.sidebar.columns(2)
if c1.button("Metric best\n(0.34 / 0.34)"):
    st.session_state.thr = 0.34
    st.session_state.nms = 0.34
    st.session_state.max_det = 300
    st.rerun()
if c2.button("Visual clean\n(0.42 / 0.34)"):
    st.session_state.thr = 0.42
    st.session_state.nms = 0.34
    st.session_state.max_det = 50
    st.rerun()

# Sliders bound to session_state (can fine-tune after pressing a preset)
thr = st.sidebar.slider("Confidence threshold", 0.00, 1.00, st.session_state.thr, 0.01, key="thr")
nms = st.sidebar.slider("NMS IoU",              0.30, 0.70, st.session_state.nms, 0.01, key="nms")
max_det = st.sidebar.slider("Max detections/image", 10, 300, st.session_state.max_det, 10, key="max_det")
show_scores = st.sidebar.checkbox("Show scores on boxes", value=True)

# ----------------------------
# Model builder (matches notebook: medium anchors, head=2 classes)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_retinanet(ckpt_path: str, device: torch.device):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    model.eval()

    # Medium anchors (start at 16 px), aspect ratios 0.5/1.0/2.0
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

    # Rebuild classification head for 1 class + background (num_classes=2)
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    in_channels = model.head.classification_head.cls_logits.in_channels
    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes=2)

    # Load trained checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nPlease place the checkpoint in the models_for_streamlit folder.")
    # load directly onto the selected device to avoid extra CPU<->GPU transfers
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)

    model.to(device)
    return model

# ----------------------------
# Utilities
# ----------------------------
to_tensor = T.ToTensor()
CLASS_ID = 1  # strawberry class id (1 class + background -> foreground is 1)

def run_inference(model, pil_img, score_thresh=0.42, nms_thresh=0.34, max_dets=50):
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
    return boxes[take], scores[take], labels[take]

def draw_boxes(pil_img, boxes, scores, show_scores=True):
    img = pil_img.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        if show_scores:
            s = float(scores[i]); txt = f"{s:.2f}"
            tw = d.textlength(txt, font=font)
            th = 16
            d.rectangle([x1, max(0, y1 - th - 2), x1 + tw + 6, y1], fill=(255, 0, 0))
            d.text((x1 + 3, max(0, y1 - th - 2)), txt, fill=(255, 255, 255), font=font)
    return img

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ----------------------------
# Load model once & show anchor sizes
# ----------------------------
load_error = None
with st.spinner("Loading RetinaNet…"):
    try:
        model = load_retinanet(CKPT_PATH, DEVICE)
        # show anchors in sidebar
        try:
            ag = model.anchor_generator
            lines = []
            for i, lvl in enumerate(ag.sizes):
                lines.append(f"P{i+3}: {tuple(int(s) for s in lvl)}")
            st.sidebar.caption("Anchor sizes:\n" + "\n".join(lines))
        except Exception:
            pass
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(load_error)
    st.stop()

# ----------------------------
# Main UI
# ----------------------------
uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"])
col_l, col_r = st.columns(2)

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    col_l.subheader("Original")
    col_l.image(pil, use_container_width=True)

    if st.button("Run Detection"):
        t0 = time.time()
        boxes, scores, labels = run_inference(model, pil, score_thresh=thr, nms_thresh=nms, max_dets=max_det)
        dt = time.time() - t0

        vis = draw_boxes(pil, boxes, scores, show_scores=show_scores)
        count = len(boxes)

        col_r.subheader(f"Detections: {count}  •  thr={thr:.2f}, nms={nms:.2f}, max_det={max_det}")
        col_r.image(vis, use_container_width=True)
        st.caption(f"Runtime: {dt:.2f}s on {DEVICE.type.upper()}")

        # Download button
        st.download_button(
            label="Download annotated image (PNG)",
            data=pil_to_bytes(vis),
            file_name=f"retinanet_pred_thr{thr:.2f}_nms{nms:.2f}.png",
            mime="image/png",
        )
else:
    st.info("Upload an image to begin. Use the sidebar to switch between metric-best and visual-clean presets.")
