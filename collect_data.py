import cv2
import os
import numpy as np

# --- 配置 ---
IMAGE_SAVE_DIR = 'images'          # 保存截图的文件夹
OUTPUT_FILE = 'annotations.txt'    # 保存标注数据的文件
WINDOW_NAME = 'Real-time Sequential Annotator'

# 定义必须按顺序标注的类别和颜色 (BGR)
CLASSES = ["black", "green", "yellow"]
COLORS = [(0, 0, 0), (0, 255, 0), (0, 255, 255)]  # 黑, 绿, 黄

# --- 全局状态变量 ---
current_class_id = 0      # 当前需要标注的类别ID (0 -> 1 -> 2)
annotations = []          # 存储当前帧的标注 [(class_id, box), ...]
drawing = False           # 标记是否正在用鼠标画框
ref_point = []            # 存储画框的坐标
saved_image_counter = 1   # 已保存图片的计数器

def find_latest_image_counter():
    """检查 'images' 文件夹，确定下一个图片的起始编号"""
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)
        return 1
    
    max_num = 0
    for f in os.listdir(IMAGE_SAVE_DIR):
        filename, _ = os.path.splitext(f)
        if filename.isdigit():
            max_num = max(max_num, int(filename))
    return max_num + 1

def draw_ui(frame):
    """在图像上绘制操作说明和当前状态"""
    h, w, _ = frame.shape
    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 动态提示需要标注的下一个物体
    if current_class_id < len(CLASSES):
        next_obj_text = f"NEXT TO DRAW: {CLASSES[current_class_id].upper()}"
        color = COLORS[current_class_id]
        cv2.rectangle(frame, (10, 10), (30, 30), color, -1)
        if current_class_id == 0: # 给黑色方块加个白边
             cv2.rectangle(frame, (10, 10), (30, 30), (255,255,255), 1)
        cv2.putText(frame, next_obj_text, (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else: # 所有物体都已标注
        all_done_text = "All 3 objects marked. Press 's' to save."
        cv2.putText(frame, all_done_text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        
    # 固定的操作说明
    instructions = "Keys: [s] Save Data | [r] Reset Frame | [q] Quit"
    cv2.putText(frame, instructions, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_annotations(frame):
    """在图像上绘制所有已完成的标注框"""
    for class_id, (x1, y1, x2, y2) in annotations:
        color = COLORS[class_id]
        label = CLASSES[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 8, y1), color, -1)
        text_color = (0, 0, 0) if class_id in [0, 2] else (255, 255, 255)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def mouse_callback(event, x, y, flags, param):
    """处理鼠标事件，并在完成一次标注后自动进入下一个类别"""
    global ref_point, drawing, annotations, current_class_id

    # 如果当前帧的3个物体都已标注完，则不再接受新的鼠标事件
    if current_class_id >= len(CLASSES):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 临时绘制，不修改全局帧
            clone = param.copy()
            draw_annotations(clone) # 先画已经确定的
            cv2.rectangle(clone, ref_point[0], (x, y), COLORS[current_class_id], 2)
            cv2.imshow(WINDOW_NAME, clone)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ref_point.append((x, y))
        
        x1, y1 = min(ref_point[0][0], ref_point[1][0]), min(ref_point[0][1], ref_point[1][1])
        x2, y2 = max(ref_point[0][0], ref_point[1][0]), max(ref_point[0][1], ref_point[1][1])

        if x2 - x1 > 5 and y2 - y1 > 5:
            annotations.append((current_class_id, (x1, y1, x2, y2)))
            # 关键步骤：自动前进到下一个类别
            current_class_id += 1

def save_data(frame_to_save):
    """保存图像和标注数据，并重置状态"""
    global annotations, current_class_id, saved_image_counter

    # 检查是否已完成所有标注
    if current_class_id < len(CLASSES):
        print("Warning: Please annotate all 3 objects (black, green, yellow) before saving.")
        return

    # 1. 保存图像
    image_filename = f"{saved_image_counter}.jpg"
    image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
    cv2.imwrite(image_path, frame_to_save)

    # 2. 保存标注
    with open(OUTPUT_FILE, 'a') as f:
        for class_id, (x1, y1, x2, y2) in annotations:
            # 注意：这里我们存的是相对路径，更通用
            line = f"{os.path.join(IMAGE_SAVE_DIR, image_filename)} {class_id} {x1} {y1} {x2} {y2}\n"
            f.write(line)
    
    print(f"--- Saved frame as {image_path} with {len(annotations)} annotations. ---")
    
    # 3. 重置状态以进行下一轮标注
    annotations = []
    current_class_id = 0
    saved_image_counter += 1
    print("System reset. Ready for next set of annotations. Please draw 'black'.")


def main():
    global saved_image_counter, annotations, current_class_id

    # 启动时确定从哪个数字开始命名文件
    saved_image_counter = find_latest_image_counter()
    print(f"Starting annotation. Images will be saved starting from '{saved_image_counter}.jpg'.")
    print("Please draw the 'black' box.")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow(WINDOW_NAME)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        
        # 创建一个用于显示的副本，避免在原始帧上绘图
        display_frame = frame.copy()
        
        # 设置鼠标回调，将显示帧的副本传进去用于临时绘制
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=display_frame.copy())
        
        # 绘制所有UI元素
        draw_annotations(display_frame)
        draw_ui(display_frame)
        
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(20) & 0xFF

        # 退出
        if key == ord('q'):
            break
        
        # 重置当前帧的标注
        elif key == ord('r'):
            annotations = []
            current_class_id = 0
            print("Annotations for this frame reset. Please start again with 'black'.")

        # 保存数据
        elif key == ord('s'):
            # 传递原始、干净的帧进行保存
            save_data(frame)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("Annotation tool closed.")

if __name__ == '__main__':
    main()