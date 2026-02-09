"use client";

import { useState, useRef } from "react";
import Searchbox from "@/app/components/Searchbox";
import { Exercise } from "@/app/data/exercises";
import WorkoutCard from "@/app/components/Workoutcard";

interface WorkoutDashboardProps {
  initialExercises: Exercise[]; // 接收从服务端传来的数据
}

export default function WorkoutDashboard({
  initialExercises,
}: WorkoutDashboardProps) {
  const categories = ["臀部", "背部", "胳膊", "腿部", "胸部", "腹部"];

  // === 状态管理 ===
  const [activeLeft, setActiveLeft] = useState(categories[0]);
  const leftContentRef = useRef<HTMLDivElement>(null);

  const [activeRight, setActiveRight] = useState(categories[0]);
  const rightContentRef = useRef<HTMLDivElement>(null);

  // 新增：搜索词状态
  const [searchText, setSearchText] = useState("");

  // === 核心功能：搜索处理函数 (自动定位) ===
  const handleSearch = (text: string) => {
    setSearchText(text); // 更新输入框文字
    
    if (!text.trim()) return; // 空字符串不处理

    const lowerText = text.toLowerCase();

    // --- 左侧查找逻辑 (Bodyweight) ---
    const leftMatch = initialExercises.find(
      (ex) => ex.type === "bodyweight" && ex.name.toLowerCase().includes(lowerText)
    );

    if (leftMatch && leftContentRef.current) {
      // 找到对应卡片的 DOM 元素
      const cardElement = document.getElementById(`card-left-${leftMatch.id}`);
      if (cardElement) {
        // 计算相对位置：元素顶部 - 容器顶部 - 20px(留白)
        const scrollPos = cardElement.offsetTop - leftContentRef.current.offsetTop - 20;
        leftContentRef.current.scrollTo({
          top: scrollPos,
          behavior: "smooth",
        });
        // 顺便更新目录高亮
        setActiveLeft(leftMatch.category);
      }
    }

    // --- 右侧查找逻辑 (Equipment) ---
    const rightMatch = initialExercises.find(
      (ex) => ex.type === "equipment" && ex.name.toLowerCase().includes(lowerText)
    );

    if (rightMatch && rightContentRef.current) {
      const cardElement = document.getElementById(`card-right-${rightMatch.id}`);
      if (cardElement) {
        const scrollPos = cardElement.offsetTop - rightContentRef.current.offsetTop - 20;
        rightContentRef.current.scrollTo({
          top: scrollPos,
          behavior: "smooth",
        });
        setActiveRight(rightMatch.category);
      }
    }
  };

  // === 交互逻辑：点击目录滚动 ===
  const scrollToCategory = (category: string, side: "left" | "right") => {
    if (side === "left") setActiveLeft(category);
    else setActiveRight(category);

    const element = document.getElementById(`${side}-${category}`);
    // 找到对应的滚动容器
    const container = side === "left" ? leftContentRef.current : rightContentRef.current;
    
    if (element && container) {
      // 使用 scrollTo 手动控制，比 scrollIntoView 更稳定，不会带着页面乱跑
      const scrollPos = element.offsetTop - container.offsetTop;
      container.scrollTo({
        top: scrollPos,
        behavior: "smooth"
      });
    }
  };

  // === 交互逻辑：滚动监听 (Scroll Spy) ===
  const handleScroll = (side: "left" | "right") => {
    const container =
      side === "left" ? leftContentRef.current : rightContentRef.current;
    if (!container) return;

    for (const cat of categories) {
      const element = document.getElementById(`${side}-${cat}`);
      if (element) {
        const rect = element.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        // 判定条件：元素顶部进入容器可视区上方 50px 到 200px 范围内
        if (
          rect.top >= containerRect.top - 50 &&
          rect.top < containerRect.top + 200
        ) {
          if (side === "left") setActiveLeft(cat);
          else setActiveRight(cat);
          break;
        }
      }
    }
  };

  // === 数据获取逻辑 (不再过滤，只按分类取) ===
  const getExercisesData = (
    category: string,
    type: "bodyweight" | "equipment",
  ) => {
    return initialExercises.filter(
      (item) => item.category === category && item.type === type,
    );
  };

  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* 搜索区域 */}
      <div className="h-[20vh] shrink-0 w-full flex flex-col items-center justify-center bg-background border-b border-white/10 relative z-10">
        <h1 className="absolute text-8xl font-black text-white/5 select-none pointer-events-none tracking-tighter">
          SEARCH
        </h1>
        {/* 关键修改：传入 value 和 onChange */}
        <Searchbox value={searchText} onChange={handleSearch} />
      </div>

      {/* 左右分栏主体 */}
      <div className="flex-1 min-h-0 grid grid-cols-2">
        {/* === 左侧区域 (Bodyweight) === */}
        <div className="flex h-full border-r border-white/10 bg-leftbox overflow-hidden">
          {/* 目录 */}
          <div className="w-24 h-full shrink-0 border-r border-white/5 flex flex-col py-6 bg-zinc-900/50 overflow-y-auto no-scrollbar">
            {categories.map((item) => (
              <button
                key={item}
                onClick={() => scrollToCategory(item, "left")}
                className={`py-4 transition-all font-bold text-sm tracking-widest text-center border-l-2 
                    ${
                      activeLeft === item
                        ? "text-lime-400 border-lime-400 bg-zinc-800"
                        : "text-zinc-500 border-transparent hover:text-zinc-300"
                    }`}
              >
                {item}
              </button>
            ))}
          </div>

          {/* 内容 */}
          <div 
            ref={leftContentRef}
            onScroll={() => handleScroll('left')}
            className="flex-1 h-full overflow-y-auto bg-transparent scroll-smooth no-scrollbar"
            style={{ direction: "rtl" }} 
          >
            <div style={{ direction: "ltr" }} className="p-4">
                {categories.map((cat) => {
                  const data = getExercisesData(cat, 'bodyweight');
                  
                  return (
                    <div key={cat} id={`left-${cat}`} className="mb-8">
                      <div className="text-zinc-600 text-xs font-bold mb-2 uppercase tracking-wider">
                        {cat}
                      </div>
                      
                      {data.length > 0 ? (
                        <div className="grid grid-cols-2 gap-4">
                           {data.map((item) => (
                              <WorkoutCard 
                                key={item.id} 
                                data={item} 
                                // 关键修改：传入 domId 以便搜索定位
                                domId={`card-left-${item.id}`}
                              />
                           ))}
                        </div>
                      ) : (
                        <div className="h-24 rounded-xl border border-dashed border-zinc-700 flex items-center justify-center text-zinc-500 text-sm">
                           该分类暂无动作
                        </div>
                      )}
                    </div>
                  )
                })}
                <div className="h-[40vh]" />
            </div>
          </div>
        </div>

        {/* === 右侧区域 (Equipment) === */}
        <div className="flex h-full bg-rightbox overflow-hidden">
          {/* 内容 */}
          <div
            ref={rightContentRef}
            onScroll={() => handleScroll("right")}
            className="flex-1 h-full overflow-y-auto p-4 scroll-smooth no-scrollbar"
          >
            {categories.map((cat) => {
              const data = getExercisesData(cat, "equipment");

              return (
                <div key={cat} id={`right-${cat}`} className="mb-8">
                  {/* 分类标题 */}
                  <div className="text-right text-zinc-600 text-xs font-bold mb-2 uppercase tracking-wider">
                    {cat}
                  </div>

                  {/* 内容区域 */}
                  {data.length > 0 ? (
                    <div className="grid grid-cols-2 gap-4">
                      {data.map((item) => (
                        <WorkoutCard 
                            key={item.id} 
                            data={item} 
                            // 关键修改：传入 domId
                            domId={`card-right-${item.id}`}
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="h-24 rounded-xl border border-dashed border-zinc-700 flex items-center justify-center text-zinc-500 text-sm">
                      该分类暂无动作
                    </div>
                  )}
                </div>
              );
            })}

            <div className="h-[40vh]" />
          </div>

          {/* 目录 */}
          <div className="w-24 h-full shrink-0 border-l border-white/5 flex flex-col py-6 bg-zinc-800/50 overflow-y-auto no-scrollbar">
            {categories.map((item) => (
              <button
                key={item}
                onClick={() => scrollToCategory(item, "right")}
                className={`py-4 transition-all font-bold text-sm tracking-widest text-center border-r-2 
                    ${
                      activeRight === item
                        ? "text-blue-400 border-blue-400 bg-zinc-700"
                        : "text-zinc-500 border-transparent hover:text-zinc-300"
                    }`}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}