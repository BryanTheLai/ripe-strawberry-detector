"""
baseline.py

Provides OpenCV template matching baseline functions for strawberry detection.
"""

import os
import logging
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple

# Configure logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def non_max_suppression(rects: List[List[int]], overlap_thresh: float = 0.3) -> np.ndarray:
    """
    Suppress overlapping bounding boxes using Non-Maximum Suppression (NMS).

    Args:
        rects: List of bounding boxes [x, y, w, h].
        overlap_thresh: Overlap threshold (IoU) for suppression.

    Returns:
        Numpy array of suppressed boxes as [[x, y, w, h], ...].
    """
    if not rects:
        return np.empty((0, 4), dtype=int)
    boxes = np.array(rects)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
    idxs = np.argsort(y2)
    pick: List[int] = []
    while idxs.size > 0:
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[last] + areas[idxs[:-1]] - inter)
        idxs = idxs[np.where(ovr <= overlap_thresh)]
    return boxes[pick].astype(int)


def create_template(xml_path: str, images_dir: str, output_path: str) -> None:
    """
    Extracts the first strawberry instance from the annotation XML and saves it as a template.

    Args:
        xml_path: Path to annotation XML.
        images_dir: Directory containing source images.
        output_path: Path to save the extracted template image.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image in root.findall('image'):
        if image.get('id') == '0':
            box = image.find('box')
            if box is None:
                logging.error("No bounding box found for image id 0.")
                return
            attrs = box.attrib
            xtl, ytl = int(float(attrs['xtl'])), int(float(attrs['ytl']))
            xbr, ybr = int(float(attrs['xbr'])), int(float(attrs['ybr']))
            img_path = os.path.join(images_dir, '0.png')
            img = cv2.imread(img_path)
            if img is None:
                logging.error(f"Failed to read image at {img_path}")
                return
            template = img[ytl:ybr, xtl:xbr]
            cv2.imwrite(output_path, template)
            logging.info(f"Template saved to {output_path}")
            return


def create_templates(xml_path: str, images_dir: str, output_dir: str, image_id: str = '0') -> None:
    """
    Crop multiple templates from all boxes of the specified image id and save under output_dir.

    Args:
        xml_path: Path to annotation XML.
        images_dir: Directory containing source images.
        output_dir: Directory to save the cropped templates.
        image_id: Image ID to filter boxes (default is '0').
    """
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image in root.findall('image'):
        if image.get('id') == image_id:
            img_path = os.path.join(images_dir, f"{image_id}.png")
            img = cv2.imread(img_path)
            if img is None:
                logging.error(f"Failed to read image at {img_path}")
                return
            for idx, box in enumerate(image.findall('box')):
                attrs = box.attrib
                xtl = int(float(attrs['xtl']))
                ytl = int(float(attrs['ytl']))
                xbr = int(float(attrs['xbr']))
                ybr = int(float(attrs['ybr']))
                templ = img[ytl:ybr, xtl:xbr]
                out_path = os.path.join(output_dir, f"template_{idx}.png")
                cv2.imwrite(out_path, templ)
            logging.info(f"Templates created in {output_dir}")
            break


def run_baseline(source_img: str, template_img: str, output_img: str, threshold: float = 0.8) -> int:
    """
    Performs template matching on a source image and saves detection visualization.

    Args:
        source_img: Path to the image to process.
        template_img: Path or directory of template images.
        output_img: Path to save the output image with detections drawn.
        threshold: Matching threshold between 0 and 1.

    Returns:
        Number of matched instances detected.
    """
    src = cv2.imread(source_img)
    if src is None:
        logging.error(f"Cannot load source image: {source_img}")
        return 0
    # Determine list of template file paths
    if os.path.isdir(template_img):
        templ_paths = [os.path.join(template_img, f) for f in os.listdir(template_img) if f.lower().endswith('.png')]
    else:
        templ_paths = [template_img]
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rects: List[List[int]] = []
    for tpath in templ_paths:
        templ = cv2.imread(tpath)
        if templ is None:
            continue
        gray_templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
        w, h = gray_templ.shape[::-1]
        res = cv2.matchTemplate(gray_src, gray_templ, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        rects.extend([[x, y, w, h] for x, y in zip(*loc[::-1])])
    picks = non_max_suppression(rects, overlap_thresh=0.3)
    for (x, y, w, h) in picks:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(output_img, src)
    count = len(picks)
    logging.info(f"Baseline detection complete - {count} items found.")
    return count
