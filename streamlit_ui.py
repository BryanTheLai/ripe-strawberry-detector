import streamlit as st
import os
import sys
import cv2
import numpy as np
import tempfile

from ultralytics import YOLO

# Add src directory to path to import our custom modules
sys.path.append(os.path.abspath('src'))

# --- Model selection UI ---
MODEL_OPTIONS = {
    'Normal YOLO (pretrained)': 'yolo11s.pt',
    'Finetuned YOLO': os.path.join('output', 'train', 'weights', 'best.pt'),
}
model_choice = st.selectbox('Select YOLO model', list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[model_choice]

# Load model
model = YOLO(MODEL_PATH)

# Determine strawberry class ID
straw_id = next((cid for cid, name in model.names.items() if name.lower() == 'strawberry'), None)

st.title('Strawberry Detector')
st.set_page_config(layout="wide")
# File uploader for images
uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

# Confidence threshold slider
conf = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)

# Run detection button
run = st.button('Run Detection')

if run:
    if uploaded_file:
        # Read uploaded image and keep original copy
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        orig = img.copy()

        with st.spinner('Detecting strawberries...'):
            # Save to temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cv2.imwrite(tmp_file.name, img)
            tmp_file.close()

            # Run YOLO inference
            results = model.predict(source=tmp_file.name, conf=conf, save=False, device='cpu')
            os.remove(tmp_file.name)

            # Extract detections
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            # Filter strawberry boxes and count
            straw_boxes = [box for box, cls in zip(boxes, classes) if cls == straw_id] if straw_id is not None else boxes.tolist()
            count = len(straw_boxes)
            # Draw and label each strawberry detection
            for idx, box in enumerate(straw_boxes, start=1):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(idx), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display original and detected images side by side
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Original Image')
            st.image(orig_rgb, use_container_width=True)
        with col2:
            st.subheader(f'Detected {count} strawberries')
            st.image(img_rgb, use_container_width=True)
    else:
        st.sidebar.warning('Please upload an image before running detection.')
