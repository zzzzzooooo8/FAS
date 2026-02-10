# server.py
# 优化后的后端入口：支持模型预加载、状态重置、实时日志

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import asyncio
from collections import deque

# 引入你的分析器和计数器
from squat_analyzer import SquatAnalyzer
from common_utils import StrictRepCounter 

# 1. 创建 FastAPI 实例
app = FastAPI()

# 2. 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 🔥 核心优化区：全局预加载 (Global Pre-loading)
# =========================================================
print("⏳ [系统启动] 正在初始化 AI 模型，这可能需要几秒钟...", flush=True)

# 在这里实例化！程序启动时只做一次！
# 这样以后前端连接时就不需要等待加载了。
GLOBAL_SQUAT_ANALYZER = SquatAnalyzer()

print("✅ [系统就绪] AI 模型加载完成！等待前端连接...", flush=True)
# =========================================================


def get_analyzer_and_reset(action_name):
    """
    工厂函数：获取全局实例，并重置它的状态（计数器归零）
    """
    if action_name == "squat":
        analyzer = GLOBAL_SQUAT_ANALYZER
        
        # ⚠️ 关键步骤：复用实例前，必须“洗碗” (重置状态)
        # 1. 重置计数器 (归零)
        analyzer.counter = StrictRepCounter()
        
        # 2. 清空 LSTM 的时序缓存 (清空之前的动作记忆)
        analyzer.frame_queue.clear()
        analyzer.result_buffer.clear()
        
        # 3. 重置反馈语
        analyzer.last_rep_feedback = "Ready"
        analyzer.current_count = 0 # 确保父类状态同步
        
        print(f"♻️ [状态重置] 已重置 {action_name} 计数器与缓存", flush=True)
        return analyzer
    else:
        return None

# 4. WebSocket 路由
@app.websocket("/ws/{action_type}")
async def websocket_endpoint(websocket: WebSocket, action_type: str):
    await websocket.accept()
    print(f"🔗 [连接成功] 前端已接入: {action_type}", flush=True)
    
    # 获取并重置分析器
    analyzer = get_analyzer_and_reset(action_type)
    
    if not analyzer:
        print(f"❌ [错误] 找不到动作 {action_type} 的分析器", flush=True)
        await websocket.close()
        return

    try:
        while True:
            # A. 接收前端发来的数据
            data = await websocket.receive_text()
            
            # B. 解码图片 (Base64 -> OpenCV)
            # 优化：快速切片，避免 split 生成大列表
            if data.startswith("data:image"):
                _, data = data.split(",", 1)
            
            image_bytes = base64.b64decode(data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None: 
                continue

            # C. 调用算法
            processed_frame, status, count, feedback = analyzer.process_frame(frame)

            # D. 编码回传 (OpenCV -> Base64)
            # 优化：使用 .jpg 而不是 .png，体积更小，传输更快
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            processed_base64 = base64.b64encode(buffer).decode('utf-8')

            # E. 发送 JSON
            await websocket.send_json({
                "image": f"data:image/jpeg;base64,{processed_base64}",
                "status": status,
                "count": count,
                "feedback": feedback
            })
            
            # F. 让出控制权 (防止 CPU 密集型任务卡死 WebSocket 心跳)
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print(f"👋 [连接断开] 前端已离开: {action_type}", flush=True)
    except Exception as e:
        print(f"❌ [系统异常] {e}", flush=True)
        # 打印详细错误堆栈，方便调试
        import traceback
        traceback.print_exc()