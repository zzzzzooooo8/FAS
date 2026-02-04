import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# --- 请修改这里 ---
VIDEO_PATH = r"dataset_raw\class_0_standard\1月21日 (4)(2).mp4"  # 确保路径完全正确！
MODEL_PATH = "yolov8n-pose.pt"

def get_best_person(results):
    """只选面积最大的人，防止选成路人"""
    if not results or len(results) == 0: return None
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0: return None
    areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
    max_index = np.argmax(areas.cpu().numpy())
    return results[0].keypoints[max_index]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_squat_curve_debug():
    # 1. 检查文件是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误：找不到文件 {VIDEO_PATH}，请检查路径！")
        return

    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    angles = []
    frames = []
    frame_idx = 0
    valid_count = 0

    print("\n--- 开始逐帧分析 ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 为了速度，只处理部分帧（调试用）
        # if frame_idx % 2 != 0: 
        #    frame_idx += 1
        #    continue

        results = model(frame, verbose=False)
        
        # 使用“最大面积”策略选人
        keypoints = get_best_person(results)
        
        if keypoints is None:
            print(f"帧 {frame_idx}: ⚠️ 未检测到任何人")
            frame_idx += 1
            continue

        # 获取数据
        kpts = keypoints.xyn.cpu().numpy()[0]
        confs = keypoints.conf.cpu().numpy()[0]

        # 这里的 None 检查很重要
        if confs is None:
            confs = np.zeros(17)

        # 智能侧面判断
        # 左侧权重 vs 右侧权重
        left_score = confs[5] + confs[11] + confs[13] # 左肩+左髋+左膝
        right_score = confs[6] + confs[12] + confs[14] # 右肩+右髋+右膝

        if left_score > right_score:
            side = "Left"
            p_shoulder, p_hip, p_knee = kpts[5], kpts[11], kpts[13]
            avg_conf = left_score / 3
        else:
            side = "Right"
            p_shoulder, p_hip, p_knee = kpts[6], kpts[12], kpts[14]
            avg_conf = right_score / 3

        # --- 调试打印 ---
        # 如果置信度太低，打印出来看看是多少
        if avg_conf < 0.2: # 我把阈值降到了 0.2，非常低了
            print(f"帧 {frame_idx}: ❌ 置信度过低 ({avg_conf:.2f}) - {side}侧")
        elif np.any(p_hip == 0):
            print(f"帧 {frame_idx}: ❌ 关键点丢失 (坐标为0)")
        else:
            # 数据有效！
            angle = calculate_angle(p_shoulder, p_hip, p_knee)
            angles.append(angle)
            frames.append(frame_idx)
            valid_count += 1
            # 每隔 10 帧打印一次成功的消息，证明在工作
            if valid_count % 10 == 0:
                print(f"帧 {frame_idx}: ✅ 成功提取 - 角度 {angle:.1f}°")

        frame_idx += 1

    cap.release()

    print(f"\n--- 分析结束 ---")
    print(f"总帧数: {frame_idx}")
    print(f"有效提取帧数: {valid_count}")

    if valid_count == 0:
        print("❌ 没有任何一帧通过筛选，无法画图。请查看上面的日志找原因。")
        return

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(frames, angles, label='Hip Angle', color='blue', linewidth=2)
    
    # 标注最低点
    min_angle = min(angles)
    min_idx = angles.index(min_angle)
    plt.scatter(frames[min_idx], min_angle, color='red', s=100, zorder=5)
    plt.text(frames[min_idx], min_angle - 5, f'Min: {int(min_angle)}°', 
             horizontalalignment='center', color='red', fontweight='bold')

    plt.title(f'Squat Analysis: {os.path.basename(VIDEO_PATH)}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (Deg)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    analyze_squat_curve_debug()