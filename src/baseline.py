import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

def non_max_suppression(rects, overlapThresh=0.3):
    if len(rects) == 0:
        return []
    boxes = np.array(rects)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]
    areas = (boxes[:,2] + 1) * (boxes[:,3] + 1)
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
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
        idxs = idxs[np.where(ovr <= overlapThresh)]
    return boxes[pick].astype(int)
 
def create_template(xml_path: str, images_dir: str, output_path: str):
    # Parse XML and get first strawberry box for image id 0
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image in root.findall('image'):
        if image.get('id') == '0':
            box = image.find('box')
            if box is None:
                raise ValueError('No bounding box found for image 0')
            # use attrib to directly access attributes
            attrs = box.attrib
            xtl = int(float(attrs['xtl']))
            ytl = int(float(attrs['ytl']))
            xbr = int(float(attrs['xbr']))
            ybr = int(float(attrs['ybr']))
            img_path = os.path.join(images_dir, '0.png')
            img = cv2.imread(img_path)
            template = img[ytl:ybr, xtl:xbr]
            cv2.imwrite(output_path, template)
            return


def create_templates(xml_path: str, images_dir: str, output_dir: str, image_id: str = '0'):
    """
    Crop multiple templates from all boxes of the specified image id and save under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image in root.findall('image'):
        if image.get('id') == image_id:
            img_path = os.path.join(images_dir, f"{image_id}.png")
            img = cv2.imread(img_path)
            for idx, box in enumerate(image.findall('box')):
                attrs = box.attrib
                xtl = int(float(attrs['xtl']))
                ytl = int(float(attrs['ytl']))
                xbr = int(float(attrs['xbr']))
                ybr = int(float(attrs['ybr']))
                templ = img[ytl:ybr, xtl:xbr]
                out_path = os.path.join(output_dir, f"template_{idx}.png")
                cv2.imwrite(out_path, templ)
            break


def run_baseline(source_img: str, template_img: str, output_img: str, threshold: float = 0.8):
    # Load source image
    src = cv2.imread(source_img)
    # Determine list of template files
    if os.path.isdir(template_img):
        templ_paths = [os.path.join(template_img, f) for f in os.listdir(template_img) if f.lower().endswith('.png')]
    else:
        templ_paths = [template_img]

    # Convert to grayscale
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rects = []
    # Match each template
    for tpath in templ_paths:
        templ = cv2.imread(tpath)
        gray_templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
        w, h = gray_templ.shape[::-1]
        res = cv2.matchTemplate(gray_src, gray_templ, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            rects.append([pt[0], pt[1], w, h])

    # Apply non-max suppression
    pick = non_max_suppression(rects, overlapThresh=0.3)

    # Draw detections
    for (x, y, w, h) in pick:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_img, src)
    print(f"Baseline - Found {len(pick)} items.")
