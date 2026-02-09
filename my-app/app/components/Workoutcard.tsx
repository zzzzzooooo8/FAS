import React from "react";
import Image from "next/image";
import { Exercise } from "@/app/data/exercises";

interface ExerciseCardProps {
  data: Exercise;
  onClick?: () => void;
  domId?: string;
}

export default function WorkoutCard({ data, onClick, domId }: ExerciseCardProps) {
  // 1. 定义动态颜色：根据类型决定 悬停时的玻璃边框颜色
  const glassBorderColor =
    data.type === "bodyweight"
      ? "group-hover:bg-lime-400/30 group-hover:border-lime-400/50" // 左边：绿色玻璃
      : "group-hover:bg-blue-400/30 group-hover:border-blue-400/50"; // 右边：蓝色玻璃

  // 2. 定义文字颜色
  const textColor = data.type === "bodyweight" ? "text-lime-400" : "text-blue-400";

  return (
    <div
      id={domId}
      onClick={onClick}
      // === 外层容器 (磨砂玻璃相框) ===
      // p-1.5: 设置边框的厚度
      // backdrop-blur-sm: 核心！实现磨砂模糊效果
      // bg-white/5: 玻璃原本的半透明底色
      className={`
        group relative aspect-square p-2 cursor-pointer transition-all duration-300
        rounded-2xl border border-white/10
        bg-white/40 backdrop-blur-md
        ${glassBorderColor}
      `}
    >
      {/* === 内层容器 (图片区域) === */}
      {/* rounded-xl: 这里的圆角要比外层稍微小一点点，视觉才协调 */}
      <div className="relative w-full h-full rounded-xl overflow-hidden bg-zinc-900">
        
        {/* 图片组件 */}
        {data.image ? (
          <Image
            src={data.image}
            alt={data.name}
            fill // 自动填满父容器
            className="object-cover transition-transform duration-500 group-hover:scale-110" // 悬停时图片轻微放大
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
        ) : (
          // 如果没有图片，显示一个默认的渐变背景兜底
          <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-900" />
        )}

        {/* === 黑色遮罩 === */}
        {/* 为了保证文字在任何图片上都清晰可见，我们在底部加一层淡淡的黑色渐变 */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60" />

        {/* === 文字区域 (固定在左下角) === */}
        <div className="absolute bottom-0 left-0 w-full p-3 flex flex-col items-start">
          
          {/* 难度标签 (胶囊样式) */}
          <div className="mb-1 px-2 py-0.5 rounded-full bg-white/20 backdrop-blur-md border border-white/10 text-[10px] text-white font-medium">
            {data.difficulty}
          </div>

          {/* 动作名称 */}
          {/* leading-tight 防止两行字间距太宽 */}
          <h3 className={`font-black text-lg leading-tight drop-shadow-md ${textColor} transition-colors`}>
            {data.name}
          </h3>
          
        </div>
      </div>
    </div>
  );
}