import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm  # 进度条库

# 1. 加载模型
model = YOLO('yolo26n-pose.pt')

def get_best_person(results):
    """
    从检测结果中挑选最合适的人（面积最大的那个）。
    防止背景里的路人干扰数据。
    """
    if not results or len(results) == 0:
        return None
    
    # 获取所有的 boxes (检测框)
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0: # 稍微增强了空值判断
        return None

    # 计算每个框的面积 (宽 x 高)
    areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
    
    # 找到面积最大的那个框的索引
    max_index = np.argmax(areas.cpu().numpy())
    
    # 返回这个人的关键点对象
    return results[0].keypoints[max_index]

def normalize_frame_count(data, target_frames=30):
    """
    均匀采样：无论视频多长，都均匀提取 target_frames 帧。
    """
    total_frames = len(data)
    if total_frames < target_frames:
        return None  # 视频太短，丢弃
    
    # 生成均匀的索引
    indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    return np.array(data)[indices]

def process_videos(raw_dir, output_dir, target_frames=30):
    # 支持的视频格式
    valid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if not os.path.exists(raw_dir):
        print(f"错误：找不到输入文件夹 {raw_dir}")
        return

    # 获取类别文件夹
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(raw_dir, class_name)
        save_path = os.path.join(output_dir, class_name)
        os.makedirs(save_path, exist_ok=True)

        video_files = [f for f in os.listdir(class_path) if os.path.splitext(f)[1].lower() in valid_exts]
        
        print(f"\n正在处理分类: {class_name}, 共 {len(video_files)} 个视频")

        # 使用 tqdm 显示进度条
        for video_name in tqdm(video_files):
            video_path = os.path.join(class_path, video_name)
            cap = cv2.VideoCapture(video_path)
            
            raw_video_points = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # 预测
                results = model(frame, verbose=False)
                
                # 获取面积最大的人
                keypoints = get_best_person(results)
                
                # === 修改核心：提取坐标 + 置信度 ===
                if keypoints is not None:
                    # 1. 获取归一化坐标 x,y -> shape: (17, 2)
                    # 注意：YOLO返回的是batch形式，单人需加 [0]
                    points = keypoints.xyn.cpu().numpy()[0]
                    
                    # 2. 获取置信度 conf -> shape: (17,)
                    confs = keypoints.conf.cpu().numpy()[0]
                    
                    # 3. 检查置信度是否存在 (有时候检测不到会返回 None)
                    if confs is None:
                        # 如果没有置信度，给个默认值 (虽然YOLOv8 pose通常都会有)
                        confs = np.ones((17, 1)) 
                    else:
                        # 将 (17,) 变形为 (17, 1) 以便拼接
                        confs = confs.reshape(17, 1)

                    # 4. 拼接 -> shape: (17, 3)
                    # 每一行变成 [x, y, conf]
                    combined_data = np.concatenate((points, confs), axis=1)
                    
                    # 5. 拉平 -> shape: (51,)
                    raw_video_points.append(combined_data.flatten())
                # === 修改结束 ===
            
            cap.release()
            
            # 均匀采样处理
            if len(raw_video_points) >= target_frames:
                final_data = normalize_frame_count(raw_video_points, target_frames)
                
                # 保存为 .npy
                save_name = os.path.splitext(video_name)[0] + ".npy"
                np.save(os.path.join(save_path, save_name), final_data)
            else:
                # 视频太短或没检测到人，跳过不保存
                pass

if __name__ == "__main__":
    # 确保这两个文件夹名字和你实际的一样
    process_videos("dataset_raw", "dataset_processed")