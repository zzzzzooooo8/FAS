# squat_analyzer.py
import torch
import numpy as np
from base_analyzer import BaseAnalyzer

# 从你的工具库导入深蹲专用的函数和类
from common_utils import (
    ActionModel, 
    StrictRepCounter, 
    get_knee_angles, 
    is_standing_pose, 
    CONF_THRESHOLD, 
    STANDING_THRESHOLD
)

class SquatAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__() # 初始化父类的 YOLO 和防抖器
        
        # 1. 加载深蹲专用模型
        # 注意：这里路径填你自己的 .pth 文件路径
        self.model_path = "best_BiLSTM.pth" 
        self.classes = ["Standard", "Butt Wink", "Knees In", "Idle"]
        
        # 初始化 BiLSTM 网络结构
        self.model = ActionModel(
            input_size=51, 
            hidden_size=128, 
            num_layers=2, 
            num_classes=4, 
            bidirectional=True
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ 深蹲模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 深蹲模型加载失败: {e}")

        # 2. 初始化深蹲专用计数器
        self.counter = StrictRepCounter()

    def predict_status(self, raw_data, frame_queue):
        """
        实现父类的 predict_status 接口
        这里包含深蹲特有的逻辑：比如 "封杀 Idle" 和 "角度过滤"
        """
        # 计算辅助角度 (用于逻辑锁)
        ang_l, ang_r = get_knee_angles(raw_data)
        valid_angs = [a for a in [ang_l, ang_r] if a != -1]
        debug_knee_angle = int(min(valid_angs)) if valid_angs else 180
        
        # 逻辑锁 1: 如果人站着，强制状态为 Idle
        if is_standing_pose(ang_l, ang_r, threshold=STANDING_THRESHOLD):
            return "Idle"

        # 准备输入数据
        input_tensor = torch.tensor(np.array(frame_queue), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(input_tensor)
            probs = torch.softmax(out, dim=1)
            
            # --- 深蹲特有的黑科技逻辑 (来自你原来的代码) ---
            probs[0, 3] = 0.0 # 强制封杀模型输出的 'Idle' 类，完全由逻辑锁控制
            
            # 如果膝盖角度还很大 (>115度)，不太可能发生屁股眨眼，强制封杀 'Butt Wink'
            if debug_knee_angle > 115: 
                probs[0, 1] = 0.0 
            # -------------------------------------------

            score, pred_idx = torch.max(probs, 1)
            raw_conf = score.item()
            
            if raw_conf > CONF_THRESHOLD:
                return self.classes[pred_idx.item()]
            else:
                return "Unsure"

    def update_counter(self, status, raw_data):
        """
        实现父类的 update_counter 接口
        使用 StrictRepCounter 进行计数
        """
        # 同样需要计算角度传给计数器
        ang_l, ang_r = get_knee_angles(raw_data)
        valid_angs = [a for a in [ang_l, ang_r] if a != -1]
        debug_knee_angle = int(min(valid_angs)) if valid_angs else 180
        
        # 调用计数器
        feedback = self.counter.process(debug_knee_angle, status)
        
        # 同步父类的计数变量 (方便 server.py 读取)
        self.current_count = self.counter.standard_count
        
        return feedback