import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm  # 进度条库

# --- 配置区域 ---
INPUT_ROOT = "dataset_raw"       # 你的原始视频根目录
OUTPUT_ROOT = "dataset_visualized" # 生成的可视化视频存放位置
MODEL_PATH = "yolov8n-pose.pt"   # 使用的 YOLO 模型

def process_and_visualize():
    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 2. 检查输入目录是否存在
    if not os.path.exists(INPUT_ROOT):
        print(f"错误: 找不到输入目录 '{INPUT_ROOT}'")
        return

    # 3. 遍历所有类别文件夹
    # 获取 class_0, class_1 等文件夹
    classes = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    for class_name in classes:
        input_class_dir = os.path.join(INPUT_ROOT, class_name)
        output_class_dir = os.path.join(OUTPUT_ROOT, class_name)
        
        # 创建对应的输出文件夹
        os.makedirs(output_class_dir, exist_ok=True)

        # 获取该类别下的所有视频文件
        video_files = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"\n>>> 正在处理类别: {class_name} (共 {len(video_files)} 个视频)")

        # 使用 tqdm 显示进度条
        for video_name in tqdm(video_files):
            input_video_path = os.path.join(input_class_dir, video_name)
            output_video_path = os.path.join(output_class_dir, video_name)

            # --- 视频处理核心逻辑 ---
            cap = cv2.VideoCapture(input_video_path)
            
            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 初始化视频写入器 (mp4v 编码兼容性较好)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO 推理
                # verbose=False 不在终端打印每一帧的信息，保持清爽
                results = model(frame, verbose=False)

                # 可视化：直接调用 plot() 方法把框和骨骼画在图上
                annotated_frame = results[0].plot()

                # 写入新视频
                out.write(annotated_frame)

            # 释放资源
            cap.release()
            out.release()

    print(f"\n✅ 全部处理完成！请前往 '{OUTPUT_ROOT}' 查看结果。")

if __name__ == "__main__":
    process_and_visualize()