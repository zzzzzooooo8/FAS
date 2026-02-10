# base_analyzer.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# 从你的工具库导入通用组件
# 假设你已经把原来的 apply.py 改名为 common_utils.py
from common_utils import AnchorStabilizer, draw_skeleton

class BaseAnalyzer:
    def __init__(self):
        """
        初始化：加载设备、YOLO、防抖器
        注意：这里不加载具体的 BiLSTM 模型，因为每个动作的模型路径不一样
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Base Analyzer on {self.device}...")
        
        # 加载通用骨架提取模型
        self.yolo = YOLO('yolov8n-pose.pt')
        
        # 初始化通用数据容器
        self.stabilizer = AnchorStabilizer(alpha=0.6)
        self.frame_queue = deque(maxlen=30)  # LSTM 需要 30 帧
        self.result_buffer = deque(maxlen=5) # 结果平滑
        
        # 计数器状态（由子类更新）
        self.current_count = 0
        self.last_feedback = "Ready"

    def predict_status(self, raw_data, frame_queue):
        """
        【抽象方法】具体的推理逻辑，由子类实现
        比如：深蹲需要过滤 Butt Wink，其他动作不需要
        """
        raise NotImplementedError("子类必须实现 predict_status 方法")

    def update_counter(self, status, raw_data):
        """
        【抽象方法】具体的计数逻辑，由子类实现
        """
        raise NotImplementedError("子类必须实现 update_counter 方法")

    def process_frame(self, frame):
        """
        通用处理流：YOLO -> 骨架提取 -> 防抖 -> (子类推理) -> (子类计数) -> 画图
        """
        # 1. YOLO 提取骨架
        results = self.yolo(frame, verbose=False, device=self.device)
        
        main_person_idx = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            main_person_idx = torch.argmax(areas).item()
        
        current_status = "Scanning..."
        box_color = (200, 200, 200) # 灰色

        if main_person_idx is not None:
            kpts = results[0].keypoints[main_person_idx]
            pts = kpts.xyn.cpu().numpy()[0]   
            confs = kpts.conf.cpu().numpy()[0].reshape(17, 1)
            # 拼接坐标和置信度 (17, 3)
            raw_data = np.concatenate((pts, confs), axis=1)

            # 2. 数据防抖处理 (通用)
            processed = self.stabilizer.process(raw_data)
            self.frame_queue.append(processed)

            # 3. 调用子类的推理逻辑 (多态)
            # 只有攒够 30 帧才开始推理
            if len(self.frame_queue) == 30:
                pred_label = self.predict_status(raw_data, self.frame_queue)
                
                # 结果平滑
                if pred_label not in ["Collecting...", "Unsure", "Idle_Fixed"]:
                    self.result_buffer.append(pred_label)
                
                if len(self.result_buffer) > 0:
                    current_status = max(set(self.result_buffer), key=self.result_buffer.count)
            else:
                current_status = "Collecting..."

            # 4. 调用子类的计数逻辑 (多态)
            feedback = self.update_counter(current_status, raw_data)
            if feedback:
                self.last_feedback = feedback

            # 5. 决定画图颜色
            if current_status == "Standard": box_color = (0, 255, 0) # 绿
            elif current_status == "Idle": box_color = (255, 255, 0) # 黄
            elif current_status in ["Butt Wink", "Knees In", "Back Sagging"]: box_color = (0, 0, 255) # 红
            
            # 6. 画骨架 (通用)
            draw_skeleton(frame, raw_data, color=box_color)

        return frame, current_status, self.current_count, self.last_feedback