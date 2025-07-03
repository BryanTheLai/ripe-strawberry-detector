import streamlit as st
import os
import sys
import cv2
import numpy as np
import tempfile
import torch

from ultralytics import YOLO

# Add src directory to path to import our custom modules
sys.path.append(os.path.abspath('src'))

st.set_page_config(layout="wide")
st.title("Strawberry Detector Video")

# Device selection (CPU/GPU)
device_options = ['cpu']
if torch.cuda.is_available():
    device_options.append('cuda:0')
device_choice = st.selectbox('Compute device', device_options)

# --- Model selection UI ---
MODEL_OPTIONS = {
    'Normal YOLO (pretrained)': 'yolo11s.pt',
    'Finetuned YOLO': os.path.join('output', 'train', 'weights', 'best.pt'),
}
model_choice = st.selectbox('Select YOLO model', list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[model_choice]

# Upload video file
uploaded_video = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mov', 'mkv'])

# Confidence threshold slider
conf = st.slider('Confidence threshold', 0.0, 1.0, 0.5, 0.05)

# Run detection button
run = st.button('Run Video Detection')

if run:
    if uploaded_video:
        # Save uploaded video to temp file
        tmp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tmp_vid.write(uploaded_video.read())
        tmp_vid.flush()
        tmp_vid.close()

        # Load model
        model = YOLO(MODEL_PATH)
        straw_id = next((cid for cid, name in model.names.items() if name.lower() == 'strawberry'), None)

        # Prepare video capture and writer
        cap = cv2.VideoCapture(tmp_vid.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        # Close temp file handle to avoid locking issues
        out_tmp_path = out_tmp.name
        out_tmp.close()
        # Use H.264 codec for browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_tmp_path, fourcc, fps, (width, height))

        with st.spinner('Processing video, please wait...'):
            # Manually read frames and run inference to preserve original resolution
            cap = cv2.VideoCapture(tmp_vid.name)
            idx_global = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Run inference on the raw frame
                results = model.predict(frame, conf=conf, save=False, device=device_choice)
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                # Annotate strawberries and label sequential indices
                count = 0
                for idx, (box, cls) in enumerate(zip(boxes, classes), start=1):
                    if cls == straw_id:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, str(idx), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        count += 1
                # Overlay count on bottom-left
                cv2.putText(frame, f'Count: {count}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                writer.write(frame)
            cap.release()

        # Release resources
        writer.release()
        cap.release()
        os.remove(tmp_vid.name)

        # Load processed video bytes and cleanup
        with open(out_tmp_path, 'rb') as f:
            video_bytes = f.read()
        os.remove(out_tmp_path)

        # Display annotated video with proper format
        st.subheader('Detection Result')
        st.video(data=video_bytes, format='video/mp4')

        # Provide download option
        base_name = os.path.splitext(uploaded_video.name)[0]
        st.download_button(
            label='Download Annotated Video',
            data=video_bytes,
            file_name=f'{base_name}_annotated.mp4',
            mime='video/mp4'
        )
    else:
        st.warning('Please upload a video before running detection.')
