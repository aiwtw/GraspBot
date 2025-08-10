import os
from PIL import Image
import random
import shutil
from pathlib import Path

def convert_annotation_line(line, img_folder, label_folder):
    parts = line.strip().split()
    img_path = parts[0].replace("\\", "/")
    cls = int(parts[1])
    x1, y1, x2, y2 = list(map(int, parts[2:]))

    img_full_path = os.path.join(img_folder, os.path.basename(img_path))
    img = Image.open(img_full_path)
    w, h = img.size

    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    label_path = os.path.join(label_folder, label_filename)
    with open(label_path, "a") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

random.seed(42)

annotation_file = "annotations.txt"
image_dir = Path("dataset/images")
label_dir = Path("dataset/labels")

os.makedirs(label_dir, exist_ok=True)
with open(annotation_file, "r") as f:
    for line in f:
        convert_annotation_line(line, image_dir, label_dir)

image_files = [f.stem for f in image_dir.glob("*.jpg")]

random.shuffle(image_files)

split_idx = int(0.9 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

for subset in ['train', 'val']:
    os.makedirs(f'dataset/images/{subset}', exist_ok=True)
    os.makedirs(f'dataset/labels/{subset}', exist_ok=True)

def move_files(file_list, subset):
    for name in file_list:
        shutil.copy(image_dir / f"{name}.jpg", f"dataset/images/{subset}/{name}.jpg")
        shutil.copy(label_dir / f"{name}.txt", f"dataset/labels/{subset}/{name}.txt")

move_files(train_files, "train")
move_files(val_files, "val")

print(f"划分完成：训练集 {len(train_files)} 张，测试集 {len(val_files)} 张。")

num_classes = 3
class_names = ['black', 'green', 'yellow']

with open("dataset/data.yaml", "w") as f:
    f.write("train: dataset/images/train\n")
    f.write("val: dataset/images/val\n")
    f.write(f"nc: {num_classes}\n")
    f.write(f"names: {class_names}\n")
