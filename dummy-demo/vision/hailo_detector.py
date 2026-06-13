"""
Hailo-8 物体检测器
==================

在 Pi5 + Hailo-8 上运行 YOLO 模型，检测工作台上的物体。
检测后结合 HSV 颜色分类，输出带颜色标签的检测结果。

依赖:
  - hailo_platform (HailoRT Python bindings)
  - numpy, cv2

使用:
    detector = HailoDetector(model_path="/data/hailo-rpi5-examples/resources/models/hailo8/yolov6n.hef")
    detector.start()
    
    frame = camera.read_color()  # BGR numpy array
    objects = detector.detect(frame)
    # → [Detection(label="block", color="red", cx=320, cy=240, bbox=[...], confidence=0.85)]
"""

import time
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """检测到的物体"""
    label: str          # YOLO 类别名 (如 "block", "cup")
    color: str          # HSV 颜色分类 (如 "red", "blue")
    confidence: float   # 检测置信度
    cx: int             # 中心 x (像素)
    cy: int             # 中心 y (像素)
    bbox: List[int]     # [x1, y1, x2, y2] (像素)
    area: int           # bbox 面积 (像素²)
    depth_mm: float = 0 # 深度值 (mm)，如果有深度相机


# ─── HSV 颜色范围定义 ────────────────────────────────────────

# 🔧 这些参数需要现场根据光照条件调整
COLOR_RANGES = {
    "red": [
        ((0, 100, 80), (10, 255, 255)),      # 红色低段
        ((170, 100, 80), (180, 255, 255)),    # 红色高段
    ],
    "blue": [
        ((100, 100, 80), (130, 255, 255)),
    ],
    "green": [
        ((40, 80, 80), (80, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
    "orange": [
        ((10, 100, 100), (20, 255, 255)),
    ],
    "purple": [
        ((130, 80, 80), (160, 255, 255)),
    ],
}


class HailoDetector:
    """
    Hailo-8 YOLO 物体检测器 + HSV 颜色分类
    
    Args:
        model_path: HEF 模型文件路径
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值
        input_size: 模型输入尺寸 (默认 640x640)
    """
    
    def __init__(
        self,
        model_path: str = "/data/hailo-rpi5-examples/resources/models/hailo8/yolov6n.hef",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        self._vdevice = None
        self._network_group = None
        self._input_info = None
        self._output_info = None
        self._started = False
    
    def start(self) -> bool:
        """初始化 Hailo 设备和模型"""
        try:
            from hailo_platform import (
                VDevice, HEF, ConfigureParams, FormatType,
                HailoStreamInterface, InferVStreams,
                InputVStreamParams, OutputVStreamParams
            )
            
            self._FormatType = FormatType
            self._InferVStreams = InferVStreams
            self._InputVStreamParams = InputVStreamParams
            self._OutputVStreamParams = OutputVStreamParams
            
            # 加载模型
            self._hef = HEF(self.model_path)
            self._input_info = self._hef.get_input_vstream_infos()
            self._output_info = self._hef.get_output_vstream_infos()
            
            input_shape = self._input_info[0].shape
            logger.info(f"Model loaded: {self.model_path}")
            logger.info(f"  Input: {self._input_info[0].name}, shape={input_shape}")
            logger.info(f"  Output: {self._output_info[0].name}, shape={self._output_info[0].shape}")
            
            # 创建设备
            params = VDevice.create_params()
            self._vdevice = VDevice(params)
            
            # 配置网络
            cfg = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(self._hef, cfg)[0]
            
            # 预激活 network group 并创建持久化 pipeline
            ng = self._network_group
            self._ng_params = ng.create_params()
            self._activated = ng.activate(self._ng_params)
            self._activated.__enter__()
            
            inp_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            self._pipeline = InferVStreams(ng, inp_params, out_params)
            self._pipeline.__enter__()
            
            self._started = True
            logger.info("Hailo detector started ✅")
            return True
            
        except Exception as e:
            logger.error(f"Hailo init failed: {e}")
            self._started = False
            return False
    
    def stop(self):
        """释放 Hailo 资源"""
        if hasattr(self, '_pipeline') and self._pipeline:
            try:
                self._pipeline.__exit__(None, None, None)
            except:
                pass
        if hasattr(self, '_activated') and self._activated:
            try:
                self._activated.__exit__(None, None, None)
            except:
                pass
        if self._vdevice:
            del self._vdevice
            self._vdevice = None
        self._started = False
        logger.info("Hailo detector stopped")
    
    def detect(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> List[Detection]:
        """
        检测图像中的物体。
        
        Args:
            frame: BGR numpy array (任意尺寸，内部会 resize)
            depth_frame: 深度图 (uint16, mm)，可选
        
        Returns:
            检测结果列表
        """
        if not self._started:
            logger.warning("Detector not started")
            return []
        
        # 预处理
        input_tensor = self._preprocess(frame)
        
        # 推理
        raw_output = self._infer(input_tensor)
        if raw_output is None:
            return []
        
        # 后处理 (NMS 已在 HEF 中完成)
        detections = self._postprocess(raw_output, frame.shape)
        
        # HSV 颜色分类
        for det in detections:
            det.color = self._classify_color(frame, det.bbox)
            if depth_frame is not None:
                det.depth_mm = self._get_depth(depth_frame, det.cx, det.cy)
        
        logger.info(f"Detected {len(detections)} objects")
        return detections
    
    def detect_specific(self, frame: np.ndarray, color: str, 
                        depth_frame: Optional[np.ndarray] = None) -> List[Detection]:
        """检测特定颜色的物体"""
        all_dets = self.detect(frame, depth_frame)
        return [d for d in all_dets if d.color == color]
    
    # ─── 内部方法 ────────────────────────────────────────────
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """预处理: resize + pad to model input size"""
        h, w = frame.shape[:2]
        size = self.input_size
        
        # Letterbox resize (保持宽高比)
        scale = min(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to square
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR→RGB (YOLO expects RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Store padding info for coordinate recovery
        self._pad_info = (scale, pad_w, pad_h, w, h)
        
        return rgb
    
    def _infer(self, input_tensor: np.ndarray) -> Optional[list]:
        """执行 Hailo 推理 (使用持久化 pipeline)"""
        try:
            batch = input_tensor[np.newaxis, ...]  # (1, H, W, 3)
            result = self._pipeline.infer({self._input_info[0].name: batch})
            raw = result[self._output_info[0].name]
            # HailoRT NMS 输出: list[batch][class] = ndarray(N, 5)
            # 取第一个 batch
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    return raw[0]  # list of 80 ndarrays
                return raw
            return raw
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
    
    def _postprocess(self, raw_output, orig_shape: tuple) -> List[Detection]:
        """
        后处理 HailoRT NMS 输出。
        
        raw_output: list of 80 ndarrays, 每个 shape (N, 5)
            5 = [y1, x1, y2, x2, score] (归一化 0-1 相对于模型输入尺寸)
        """
        scale, pad_w, pad_h, orig_w, orig_h = self._pad_info
        input_size = self.input_size
        detections = []
        
        if not isinstance(raw_output, list):
            return []
        
        for cls_id, cls_dets in enumerate(raw_output):
            if not hasattr(cls_dets, 'shape') or cls_dets.shape[0] == 0:
                continue
            
            for det_row in cls_dets:
                score = det_row[4]
                if score < self.conf_threshold:
                    continue
                
                # HailoRT NMS 输出: [y1, x1, y2, x2, score] 归一化到 0-1
                y1_norm, x1_norm, y2_norm, x2_norm = det_row[:4]
                
                # 转换到模型输入空间 (pixel)
                x1 = x1_norm * input_size
                y1 = y1_norm * input_size
                x2 = x2_norm * input_size
                y2 = y2_norm * input_size
                
                # 去除 padding 并缩放到原图
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale
                
                # Clamp
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = int((x2 - x1) * (y2 - y1))
                
                if area < 100:  # 太小的忽略
                    continue
                
                detections.append(Detection(
                    label=self._get_class_name(cls_id),
                    color="unknown",
                    confidence=float(score),
                    cx=cx, cy=cy,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    area=area,
                ))
        
        return detections
    
    def _classify_color(self, frame: np.ndarray, bbox: List[int]) -> str:
        """通过 HSV 判断 bbox 区域内物体的颜色"""
        x1, y1, x2, y2 = bbox
        
        # 取 bbox 中心区域 (避免边缘噪声)
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        roi = frame[y1+margin_y:y2-margin_y, x1+margin_x:x2-margin_x]
        
        if roi.size == 0:
            return "unknown"
        
        # 转 HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 统计每种颜色的像素占比
        best_color = "unknown"
        best_ratio = 0.0
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                m = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, m)
            
            ratio = np.count_nonzero(mask) / total_pixels
            if ratio > best_ratio and ratio > 0.2:  # 至少 20% 像素匹配
                best_ratio = ratio
                best_color = color_name
        
        return best_color
    
    def _get_depth(self, depth_frame: np.ndarray, cx: int, cy: int) -> float:
        """从深度图中获取物体深度 (取中心区域中位数)"""
        h, w = depth_frame.shape[:2]
        # 取 5x5 邻域中位数
        r = 5
        y1 = max(0, cy - r)
        y2 = min(h, cy + r)
        x1 = max(0, cx - r)
        x2 = min(w, cx + r)
        
        patch = depth_frame[y1:y2, x1:x2]
        valid = patch[patch > 0]
        
        if len(valid) > 0:
            return float(np.median(valid))
        return 0.0
    
    def _get_class_name(self, cls_id: int) -> str:
        """COCO 类别名（只保留我们关心的）"""
        # YOLOv6n 在 COCO 80 类上训练
        # 对于方块 demo，所有检测到的物体都当作 "block"
        # 可以后续用自定义模型替换
        COCO_NAMES = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird",
            "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        ]
        if 0 <= cls_id < len(COCO_NAMES):
            return COCO_NAMES[cls_id]
        return f"class_{cls_id}"

    # ─── 上下文管理器 ────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
