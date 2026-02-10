"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Webcam from "react-webcam";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { Camera, RefreshCw, Zap, ArrowLeft, AlertCircle } from "lucide-react";
import { Exercise } from "@/app/data/exercises";

interface TrainingViewProps {
  exercise: Exercise;
}

export default function TrainingView({ exercise }: TrainingViewProps) {
  const router = useRouter();
  const webcamRef = useRef<Webcam>(null);

  // === 状态管理 ===
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [stats, setStats] = useState({
    count: 0,
    status: "Ready",
    feedback: "调整好位置，准备开始",
  });

  const [errorMsg, setErrorMsg] = useState<string>("");

  // === 1. WebSocket 连接 ===
  const socketUrl =
    isCameraOn && exercise.idName
      ? `ws://localhost:8000/ws/${exercise.idName}`
      : null;

  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl, {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
    reconnectAttempts: 10,
    onOpen: () => {
      console.log("✅ WebSocket 连接成功:", socketUrl);
      setErrorMsg("");
    },
    onClose: () => console.log("⚠️ WebSocket 连接关闭"),
    onError: (event) => {
      console.error("❌ WebSocket 错误:", event);
      setErrorMsg("无法连接到 AI 服务器，请检查后端是否启动");
    },
  });

  // === 核心逻辑：发送帧函数 (乒乓模式的球拍) ===
  const sendFrame = useCallback(() => {
    if (!isCameraOn || readyState !== ReadyState.OPEN || !webcamRef.current)
      return;

    // 获取截图 (Base64)
    const imageSrc = webcamRef.current.getScreenshot();

    if (imageSrc) {
      sendMessage(imageSrc);
    }
  }, [isCameraOn, readyState, sendMessage]);

  // === 2. 接收数据 & 触发下一次发送 (乒乓模式的核心) ===
  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const data = JSON.parse(lastMessage.data);

        // 使用 requestAnimationFrame 更新界面，避免卡顿
        requestAnimationFrame(() => {
          setProcessedImage(data.image);
          setStats({
            count: data.count,
            status: data.status,
            feedback: data.feedback,
          });
        });

        // 🔥 重点：收到回复后，立刻发送下一帧！
        // 使用 setTimeout(..., 0) 将发送操作放入宏任务队列，给浏览器 UI 线程喘息机会
        setTimeout(() => {
          sendFrame();
        }, 0);
      } catch (e) {
        console.error("解析后端数据失败:", e);
      }
    }
  }, [lastMessage, sendFrame]);

  // === 3. 启动逻辑：暴力点火 (修复一直转圈的问题) ===
  useEffect(() => {
    let startupInterval: NodeJS.Timeout;

    // 只有当：1.相机开了 2.WebSocket连上了 3.还没收到过回信(processedImage为空) 时，才启动点火
    if (isCameraOn && readyState === ReadyState.OPEN && !processedImage) {
      console.log("🚀 [启动程序] 开始尝试发送第一帧...");

      startupInterval = setInterval(() => {
        // 双重检查：如果中间突然收到图了，立马停止
        if (processedImage) {
          console.log(
            "✅ [启动成功] 已收到后端回复，停止手动发送，切换为自动乒乓模式",
          );
          clearInterval(startupInterval);
          return;
        }

        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();

          if (imageSrc) {
            console.log("📨 [发送中] 成功截取到图片，正在发送给后端...");
            sendMessage(imageSrc);
          } else {
            console.log(
              "⏳ [等待中] 摄像头正在预热，getScreenshot 返回 null...",
            );
          }
        } else {
          console.log("❌ [错误] webcamRef 为空，组件可能未加载");
        }
      }, 500); // 每 500ms 尝试一次 (比 200ms 稳一点，给摄像头喘息时间)
    }

    return () => clearInterval(startupInterval);
  }, [isCameraOn, readyState, processedImage, sendMessage]);
  // 状态颜色辅助函数
  const getStatusColor = (status: string) => {
    if (status === "Standard") return "text-lime-400 border-lime-400";
    if (status === "Idle") return "text-yellow-400 border-yellow-400";
    return "text-red-500 border-red-500";
  };

  return (
    <div className="h-screen w-full flex flex-col bg-black text-white overflow-hidden">
      {/* === Header === */}
      <div className="h-16 shrink-0 flex items-center px-6 border-b border-white/10 bg-black/50 backdrop-blur-sm z-50 relative justify-between">
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 px-4 py-2 rounded-full hover:bg-white/10 transition-colors text-sm font-medium"
        >
          <ArrowLeft size={18} />
          <span>返回</span>
        </button>

        <h1 className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xl font-bold tracking-wider uppercase">
          {exercise.name}
        </h1>

        <div className="w-16"></div>
      </div>

      {/* === 主内容区 === */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 w-full overflow-hidden relative">
        {errorMsg && (
          <div className="absolute top-4 z-50 bg-red-500/90 text-white px-6 py-2 rounded-full flex items-center gap-2 animate-pulse">
            <AlertCircle size={18} />
            <span>{errorMsg}</span>
          </div>
        )}

        <div
          className={`transition-all duration-500 ease-in-out ${isCameraOn ? "w-full max-w-6xl h-full flex gap-4" : "w-full max-w-lg aspect-square"}`}
        >
          {/* 左侧：视频区域 */}
          <div
            className={`relative flex-1 bg-zinc-900 rounded-3xl overflow-hidden border-2 ${isCameraOn ? "border-zinc-700" : "border-dashed border-zinc-700"} flex justify-center items-center transition-all`}
          >
            {!isCameraOn && (
              <button
                onClick={() => setIsCameraOn(true)}
                className="group w-full h-full flex flex-col items-center justify-center hover:bg-zinc-800/50 transition-all"
              >
                <div className="p-6 rounded-full bg-zinc-800 group-hover:bg-lime-400 transition-colors duration-300 shadow-xl mb-6">
                  <Camera className="w-10 h-10 text-zinc-400 group-hover:text-black transition-colors" />
                </div>
                <span className="text-zinc-500 font-medium group-hover:text-lime-400 transition-colors">
                  点击开启 {exercise.name} 训练
                </span>
              </button>
            )}

            {/* === 修改后的结构 === */}

            {/* 1. Webcam 放在所有条件判断外面，始终渲染！ */}
            {/* 注意：className 里用 hidden 来控制显示/隐藏，而不是销毁组件 */}
            <Webcam
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{
                width: 480,
                height: 360,
                facingMode: "user",
              }}
              screenshotQuality={0.5}
              width={640}
              height={480}
              mirrored={true}
              // 🔥 关键点：如果还没开启，就用 hidden 隐藏；开启后，用 pointer-events-none 让它作为背景
              // 同时也暂时去掉了 opacity-0，方便你调试看到它到底有没有画面
              className={`absolute z-0 ${isCameraOn ? "opacity-100" : "hidden"}`}
              // 🔥 加上这个监听，确认摄像头真的活了
              onUserMedia={() => console.log("📷 摄像头硬件已就绪！")}
              onUserMediaError={(e) => console.error("❌ 摄像头启动失败！", e)}
            />

            {/* 2. 原来的条件渲染区域，只保留处理后的图片和 Loading */}
            {isCameraOn && (
              <div className="relative z-10 w-full h-full flex items-center justify-center">
                {processedImage ? (
                  <img
                    src={processedImage}
                    className="w-full h-full object-contain animate-in fade-in"
                    alt="AI Analysis"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-3 text-zinc-500 bg-black/50 p-4 rounded-xl backdrop-blur-sm">
                    <RefreshCw className="w-8 h-8 animate-spin" />
                    <span>AI 正在接入视频流...</span>
                    <span className="text-xs text-zinc-600">
                      URL: {socketUrl}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 右侧：数据面板 */}
          {isCameraOn && (
            <div className="w-80 shrink-0 bg-zinc-900/80 backdrop-blur-xl border border-zinc-800 rounded-3xl p-6 flex flex-col justify-between animate-in slide-in-from-right-10 fade-in duration-500">
              <div>
                <h3 className="text-zinc-500 text-xs font-bold tracking-widest uppercase mb-4">
                  Analysis
                </h3>
                <div
                  className={`border-l-4 pl-4 py-2 ${getStatusColor(stats.status)} bg-zinc-800/50 rounded-r-lg`}
                >
                  <div className="text-2xl font-black italic">
                    {stats.status}
                  </div>
                  <div className="text-xs font-medium opacity-80 mt-1">
                    当前动作状态
                  </div>
                </div>
              </div>

              <div className="text-center py-4">
                <div className="relative inline-block">
                  <span className="text-9xl font-black tracking-tighter text-white drop-shadow-2xl">
                    {stats.count}
                  </span>
                  <Zap className="absolute top-0 -right-6 text-yellow-400 w-8 h-8 fill-current animate-bounce" />
                </div>
                <p className="text-zinc-500 text-xs font-bold tracking-[0.3em] mt-2">
                  REPS
                </p>
              </div>

              <div
                className={`rounded-2xl p-4 min-h-[120px] flex items-center justify-center relative overflow-hidden transition-colors duration-300 ${stats.feedback.includes("Standard") ? "bg-lime-900/20" : "bg-red-900/20"}`}
              >
                <p
                  className={`text-center font-bold text-lg leading-tight ${stats.feedback.includes("Standard") || stats.feedback.includes("Ready") ? "text-lime-400" : "text-orange-400"}`}
                >
                  {stats.feedback}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
