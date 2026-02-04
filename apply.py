import cv2
import torch
import collections
import numpy as np
from ultralytics import YOLO

# =================配置区=================
INPUT_SOURCE = r"dataset_raw\class_0_standard\1月21日 (2)(3).mp4" # 摄像头
MODEL_PATH = "best_BiLSTM.pth" 
CONF_THRESHOLD = 0.60 
STANDING_THRESHOLD = 150.0 
SQUAT_DEPTH_THRESHOLD = 110.0 
# =======================================

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- 1. 严格计数器 (Only Count Standard) ---
class StrictRepCounter:
    def __init__(self):
        self.standard_count = 0 # 只记标准数
        self.stage = "UP" 
        self.is_clean_rep = True # 标记当前这一个动作是否干净
        self.fail_reason = ""    # 记录错误原因

    def process(self, knee_angle, ai_status):
        # 1. 下蹲开始
        if knee_angle < SQUAT_DEPTH_THRESHOLD:
            if self.stage == "UP":
                self.stage = "DOWN"
                self.is_clean_rep = True # 新的一轮，先假设是好的
                self.fail_reason = ""
                print(">>> 下蹲开始...")
        
        # 2. 下蹲过程监控
        if self.stage == "DOWN":
            # 如果出现了错误动作，且置信度比较高(AI确定是错的)
            if ai_status in ["Butt Wink", "Knees In"]:
                self.is_clean_rep = False
                self.fail_reason = ai_status # 记住罪魁祸首

        # 3. 起身结算
        if knee_angle > STANDING_THRESHOLD:
            if self.stage == "DOWN":
                self.stage = "UP"
                
                # === 核心修改：只有干净的动作才计数 ===
                if self.is_clean_rep:
                    self.standard_count += 1
                    print(f">>> 标准动作！计数+1 (总数: {self.standard_count})")
                    return "Standard (+1)"
                else:
                    print(f">>> 动作错误 ({self.fail_reason})，不计数！")
                    return f"No Count ({self.fail_reason})"
        
        return None

# --- 2. 锚点稳定器 ---
class AnchorStabilizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_anchor = None
        self.prev_scale = 1.0

    def process(self, frame_data):
        data = frame_data.copy()
        ignore_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
        data[ignore_indices, :] = 0.0 
        confs = data[:, 2]

        target_anchor = None
        if confs[11] > 0.3 and confs[12] > 0.3: 
            target_anchor = (data[11, :2] + data[12, :2]) / 2
        elif confs[5] > 0.3 and confs[6] > 0.3: 
            target_anchor = (data[5, :2] + data[6, :2]) / 2
        else:
            target_anchor = self.prev_anchor if self.prev_anchor is not None else np.zeros(2)

        current_scale = self.prev_scale
        if confs[5] > 0.3 and confs[6] > 0.3 and confs[11] > 0.3 and confs[12] > 0.3:
             sh_c = (data[5, :2] + data[6, :2]) / 2
             hip_c = (data[11, :2] + data[12, :2]) / 2
             dist = np.linalg.norm(sh_c - hip_c)
             if dist > 0.05: current_scale = dist
        self.prev_scale = current_scale

        if self.prev_anchor is None: smooth_anchor = target_anchor
        else: smooth_anchor = (self.alpha * target_anchor) + ((1 - self.alpha) * self.prev_anchor)
        self.prev_anchor = smooth_anchor
        
        data[:, :2] -= smooth_anchor 
        data[:, :2] /= current_scale 
        data[~ (data[:, 2] > 0), :2] = 0 
        return data.flatten()

# --- 3. 辅助函数 ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_knee_angles(frame_data):
    pts = frame_data
    l_hip, l_knee, l_ankle = pts[11], pts[13], pts[15]
    r_hip, r_knee, r_ankle = pts[12], pts[14], pts[16]
    ang_l = calculate_angle(l_hip, l_knee, l_ankle) if l_knee[2] > 0.3 else -1
    ang_r = calculate_angle(r_hip, r_knee, r_ankle) if r_knee[2] > 0.3 else -1
    return ang_l, ang_r

def is_standing_pose(ang_l, ang_r, threshold=150):
    valid = [a for a in [ang_l, ang_r] if a != -1]
    if not valid: return False 
    if min(valid) > threshold: return True
    return False

def draw_skeleton(frame, kpts, color=(0, 255, 0)):
    h, w, _ = frame.shape
    for i, j in SKELETON_CONNECTIONS:
        pt1, pt2 = kpts[i], kpts[j]
        if pt1[2] > 0.4 and pt2[2] > 0.4:
            x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
            x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    for i, (x_n, y_n, conf) in enumerate(kpts):
        if conf > 0.4: cv2.circle(frame, (int(x_n*w), int(y_n*h)), 5, color, -1)

# --- 4. 模型定义 ---
class ActionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=bidirectional)
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = torch.nn.Linear(fc_input_dim, num_classes)
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        if self.bidirectional: out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else: out = h_n[-1,:,:]
        return self.fc(out)

# --- 主程序 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    yolo = YOLO('yolov8n-pose.pt')
    model = ActionModel(input_size=51, hidden_size=128, num_layers=2, num_classes=4, bidirectional=True).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 模型加载成功")
    except:
        print("❌ 模型加载失败")
        return
    model.eval()

    cap = cv2.VideoCapture(INPUT_SOURCE)
    frame_queue = collections.deque(maxlen=30)
    result_buffer = collections.deque(maxlen=5)
    classes = ["Standard", "Butt Wink", "Knees In", "Idle"]
    
    stabilizer = AnchorStabilizer(alpha=0.6)
    counter = StrictRepCounter() # 使用严格计数器
    
    cv2.namedWindow('Strict AI Coach', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Strict AI Coach', 1280, 720)
    
    last_rep_feedback = "Ready"
    feedback_color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = yolo(frame, verbose=False, device=device)
        
        main_person_idx = None
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            areas = results[0].boxes.xywh[:, 2] * results[0].boxes.xywh[:, 3]
            main_person_idx = torch.argmax(areas).item()
        
        current_status = "Scanning..."
        confidence_score = 0.0
        box_color = (200, 200, 200)

        if main_person_idx is not None:
            kpts = results[0].keypoints[main_person_idx]
            pts = kpts.xyn.cpu().numpy()[0]   
            confs = kpts.conf.cpu().numpy()[0].reshape(17, 1)
            raw_data = np.concatenate((pts, confs), axis=1)

            # 1. 计算角度
            ang_l, ang_r = get_knee_angles(raw_data)
            valid_angs = [a for a in [ang_l, ang_r] if a != -1]
            debug_knee_angle = int(min(valid_angs)) if valid_angs else 180
            
            # 2. 逻辑锁判断
            is_standing = is_standing_pose(ang_l, ang_r, threshold=STANDING_THRESHOLD)

            # 3. 数据预处理
            processed = stabilizer.process(raw_data)
            frame_queue.append(processed)

            # 4. 推理
            if is_standing:
                pred_label = "Idle"
                confidence_score = 1.0
            else:
                if len(frame_queue) == 30:
                    input_tensor = torch.tensor(np.array(frame_queue), dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(input_tensor)
                        probs = torch.softmax(out, dim=1)
                        
                        probs[0, 3] = 0.0 # 封杀 Idle
                        if debug_knee_angle > 115: probs[0, 1] = 0.0 # 封杀 Butt Wink

                        score, pred_idx = torch.max(probs, 1)
                        raw_conf = score.item()
                        raw_label = classes[pred_idx.item()]
                        
                        if raw_conf > CONF_THRESHOLD:
                            pred_label = raw_label
                            confidence_score = raw_conf
                        else:
                            pred_label = "Unsure"
                            confidence_score = raw_conf
                else:
                    pred_label = "Collecting..."

            # 5. 结果平滑
            if pred_label not in ["Collecting...", "Unsure"]:
                result_buffer.append(pred_label)
            if len(result_buffer) > 0:
                current_status = max(set(result_buffer), key=result_buffer.count)

            # 6. 【核心】严格计数器更新
            rep_result = counter.process(debug_knee_angle, current_status)
            if rep_result:
                last_rep_feedback = rep_result
                # 设置反馈颜色：标准绿色，错误红色
                feedback_color = (0, 255, 0) if "Standard" in rep_result else (0, 0, 255)

            # 7. 绘图 UI
            if current_status == "Standard": box_color = (0, 255, 0)
            elif current_status == "Idle": box_color = (255, 255, 0)
            elif current_status in ["Butt Wink", "Knees In"]: box_color = (0, 0, 255)
            
            draw_skeleton(frame, raw_data, color=box_color)
            
            # 左上角：计数面板
            cv2.rectangle(frame, (0, 0), (320, 180), (0, 0, 0), -1)
            cv2.putText(frame, f"Real-time: {current_status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            cv2.putText(frame, f"Standard Reps: {counter.standard_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            
            # 显示上一次动作的评价
            cv2.putText(frame, last_rep_feedback, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

            # 左下角：调试信息
            debug_info = f"Angle: {debug_knee_angle} | Stage: {counter.stage}"
            cv2.putText(frame, debug_info, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Strict AI Coach', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()