import torch
import numpy as np
import sys
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class YOLODetector:
    def __init__(
        self,
        weights=ROOT / "yolov5s.pt",  # model path
        data=ROOT / "data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        """
        Initialize the YOLOv5 detector by loading the model once.
        """
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride = self.model.stride
        self.names = self.model.names  # class names
        self.pt = self.model.pt  # PyTorch model
        
        # Check image size
        self.imgsz = check_img_size(imgsz, s=self.stride)
        
        # Warmup
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))
        
        # Save parameters for detection
        self.half = half
        
    def detect(
        self,
        img,  # OpenCV image (BGR format)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
    ):
        """
        Performs object detection on a single OpenCV image and returns detected objects.
        
        Args:
            img: OpenCV image in BGR format
            
        Returns:
            list: List of detections, each in format [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # Prepare image
        im = img.copy()
        im = im[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # Inference
        pred = self.model(im, augment=augment)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process predictions
        results = []
  # 遍历每张图片的检测结果 (通常批次为1)
        for i, det in enumerate(pred):
            if len(det):
                # 将边界框从模型推理尺寸(img_size)缩放回原始图像尺寸(im0)
                # 注意: 这里的 im.shape 是预处理后的tensor尺寸, img.shape 是原始图像尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                
                # ------------------- 修改后的核心逻辑开始 -------------------
                
                # 1. 初始化一个字典来存储每个类别的最佳检测
                #    键: class_id, 值: [x1, y1, x2, y2, confidence, class_id, class_name]
                best_detections_per_class = {}

                # 2. 遍历所有检测结果
                for *xyxy, conf, cls in det:
                    confidence = float(conf)
                    class_id = int(cls)

                    # 3. 比较和更新
                    # 如果该类别还未记录，或者当前检测的置信度更高
                    if class_id not in best_detections_per_class or confidence > best_detections_per_class[class_id][4]:
                        # 获取坐标和类别名
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        class_name = self.names[class_id]
                        
                        # 更新字典中该类别的最佳检测
                        best_detections_per_class[class_id] = [x1, y1, x2, y2, confidence, class_id, class_name]

                # 4. 从字典中提取结果，生成最终的列表
                if best_detections_per_class:
                    results = list(best_detections_per_class.values())
                
        return results
    
    def preprocess(self, img_source):
        # 将 BGR 格式的 OpenCV 图像转换为 RGB
        img = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
        # 转换为 CHW 格式
        img = img.transpose((2, 0, 1))
        # 转换为 Tensor
        img = torch.from_numpy(img).to(self.device)
        # 归一化
        img = img.float() / 255.0
        # 增加 batch 维度
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

# -------------------------------------------------------------
# 这是被修改和增强的主程序部分
# -------------------------------------------------------------
if __name__ == '__main__':
    # 初始化检测器 (只需加载一次模型)
    detector = YOLODetector(
        weights='./runs/train/exp5/weights/best.pt',  # 模型权重路径
        data='./dataset/data.yaml',                   # 数据集配置文件路径
        device=''                                     # 留空以自动选择设备 (CPU/GPU)
    )

    # 从摄像头或视频文件处理
    cap = cv2.VideoCapture(0)  # 0 代表默认摄像头, 也可以是 'your_video.mp4'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 在帧上检测目标
        detections = detector.detect(frame)
        
        # --- 这是新增的绘图部分 ---
        # 遍历所有检测到的目标
        for det in detections:
            # 解包检测结果
            x1, y1, x2, y2, confidence, class_id, class_name = det

            # 1. 绘制边界框
            # cv2.rectangle(图像, 左上角坐标, 右下角坐标, 颜色 (BGR), 线条粗细)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2. 准备标签文本
            label = f'{class_name} {confidence:.2f}'
            
            # 3. 绘制标签背景
            # 计算文本大小以创建背景
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1) # -1 表示填充

            # 4. 绘制标签文本
            # cv2.putText(图像, 文本, 左下角坐标, 字体, 字体大小, 颜色, 粗细)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # --- 绘图部分结束 ---

        cv2.imshow('YOLOv5 Real-Time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()