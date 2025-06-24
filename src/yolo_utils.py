import os
import shutil
import random
import xml.etree.ElementTree as ET

def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for image in root.findall('image'):
        img_id = image.get('id')
        name_attr = image.get('name')
        if name_attr is None:
            raise ValueError(f"Image element missing 'name' attribute: {ET.tostring(image)}")
        img_name = name_attr.split('/')[-1]
        boxes = []
        for box in image.findall('box'):
            attrs = box.attrib
            xtl = float(attrs['xtl'])
            ytl = float(attrs['ytl'])
            xbr = float(attrs['xbr'])
            ybr = float(attrs['ybr'])
            # convert to YOLO center x, center y, width, height (normalized later)
            boxes.append((xtl, ytl, xbr, ybr))
        data.append((img_name, boxes))
    return data

def convert_to_yolo_format(box, img_w, img_h):
    xtl, ytl, xbr, ybr = box
    x_center = ((xtl + xbr) / 2) / img_w
    y_center = ((ytl + ybr) / 2) / img_h
    width = (xbr - xtl) / img_w
    height = (ybr - ytl) / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def split_and_prepare(xml_path, images_dir, output_dir, train_ratio=0.8, seed=42):
    data = parse_annotations(xml_path)
    random.Random(seed).shuffle(data)
    split = int(len(data) * train_ratio)
    groups = {
        'train': data[:split],
        'val': data[split:]
    }
    for group, items in groups.items():
        img_out = os.path.join(output_dir, 'images', group)
        lbl_out = os.path.join(output_dir, 'labels', group)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_name, boxes in items:
            # copy image
            shutil.copy(os.path.join(images_dir, img_name), img_out)
            # open image to get size
            import cv2
            img = cv2.imread(os.path.join(images_dir, img_name))
            h, w = img.shape[:2]
            # write yolo file
            txt_path = os.path.join(lbl_out, img_name.replace('.png', '.txt'))
            with open(txt_path, 'w') as f:
                for box in boxes:
                    line = convert_to_yolo_format(box, w, h)
                    f.write(line + '\n')
    # write data.yaml
    yaml = f"train: images/train\n"
    yaml += f"val: images/val\n"
    yaml += f"nc: 1\n"
    yaml += "names: ['strawberry']\n"
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml)
