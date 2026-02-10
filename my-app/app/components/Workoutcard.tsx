"use client"; // 必须是客户端组件才能用路由

import React from "react";
import Image from "next/image";
import { useRouter } from "next/navigation"; // <--- 1. 引入路由钩子
import { Exercise } from "@/app/data/exercises";

interface WorkoutCardProps {
  data: Exercise;
  domId?: string;
  // onClick 可以保留，作为额外的回调，但主要跳转逻辑我们在内部处理
  onClick?: () => void; 
}

export default function WorkoutCard({ data, onClick, domId }: WorkoutCardProps) {
  const router = useRouter(); // <--- 2. 初始化路由

  // 根据类型决定边框颜色
  const glassBorderColor =
    data.type === "bodyweight"
      ? "group-hover:bg-lime-400/30 group-hover:border-lime-400/50"
      : "group-hover:bg-blue-400/30 group-hover:border-blue-400/50";

  const textColor = data.type === "bodyweight" ? "text-lime-400" : "text-blue-400";

  // === 3. 处理点击事件 ===
  const handleClick = () => {
    // 如果父组件传了 onClick (比如统计打点)，先执行它
    if (onClick) onClick();
    
    // 核心跳转逻辑！去往 /train/动作ID
    router.push(`/train/${data.idName}`);
  };

  return (
    <div
      id={domId}
      onClick={handleClick} // <--- 4. 绑定点击事件
      className={`
        group relative aspect-square p-2 cursor-pointer transition-all duration-300
        rounded-2xl border border-white/10
        bg-white/5 backdrop-blur-md
        ${glassBorderColor}
      `}
    >
      {/* ... 内部代码保持不变 ... */}
      <div className="relative w-full h-full rounded-xl overflow-hidden bg-zinc-900">
        {data.image && (data.image.startsWith('/') || data.image.startsWith('http')) ? (
          <Image
            src={data.image}
            alt={data.name}
            fill
            className="object-cover transition-transform duration-500 group-hover:scale-110"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-zinc-800 to-zinc-900" />
        )}

        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60" />

        <div className="absolute bottom-0 left-0 w-full p-3 flex flex-col items-start">
          <div className="mb-1 px-2 py-0.5 rounded-full bg-white/20 backdrop-blur-md border border-white/10 text-[10px] text-white font-medium">
            {data.difficulty}
          </div>
          <h3 className={`font-black text-lg leading-tight drop-shadow-md ${textColor} transition-colors`}>
            {data.name}
          </h3>
        </div>
      </div>
    </div>
  );
}