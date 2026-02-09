// app/components/MuscleTrainingFeature.tsx
'use client'; // 👈 只有这里需要标记为客户端组件

import { useState } from 'react';
import HumanCanvas from '@/app/components/HumanCanvas';
import { getMuscleInfo, MuscleData } from '@/app/data/muscles';

export default function MuscleTrainingFeature() {
  // === 这里的逻辑原本在 page.tsx 里 ===
  const [activeMuscle, setActiveMuscle] = useState<MuscleData | null>(null);

  const handleMuscleSelect = (meshName: string) => {
    const info = getMuscleInfo(meshName);
    setActiveMuscle(info);
  };

  return (
    <div className="flex w-full h-full"> {/* 注意：这里的高度由父级控制 */}
      
      {/* === 左侧：3D 模型区域 === */}
      <div className="flex-1 relative h-full">
        <HumanCanvas onMuscleSelect={handleMuscleSelect} />
        
        {/* 提示层 */}
        <div className="absolute top-24 left-4 bg-white/70 p-4 rounded-lg backdrop-blur-sm z-10 pointer-events-none">
          <h1 className=" !text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            3D 肌肉解剖
          </h1>
          <p className="text-gray-400 text-sm mt-1">点击模型查看训练动作</p>
        </div>
      </div>

      {/* === 右侧：训练动作面板 === */}
      <div className="w-80 bg-leftbox border-lshadow-2xl flex flex-col transition-all z-20">
        <div className="p-6 border-b border-gray-700">
          <h2 className="mt-8 text-xl font-bold flex items-center gap-2 text-white">
            🏋️ 部位训练
          </h2>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {activeMuscle ? (
            <div className="animate-in slide-in-from-right duration-300">
              <h3 className="text-3xl font-extrabold text-blue-400 mb-2">
                {activeMuscle.name}
              </h3>
              
              <div className="space-y-6 mt-6">
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
                    推荐训练动作
                  </h4>
                  <ul className="space-y-3">
                    {activeMuscle.actions.length > 0 ? (
                      activeMuscle.actions.map((action, i) => (
                        <li key={i} className="group p-3 bg-gray-700/50 rounded-xl hover:bg-gray-700 transition-all cursor-pointer border border-transparent hover:border-blue-500/30">
                          <div className="flex items-center gap-3">
                            <span className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm font-bold group-hover:bg-blue-500 group-hover:text-white transition-colors">
                              {i + 1}
                            </span>
                            <span className="font-medium text-gray-100">{action}</span>
                          </div>
                        </li>
                      ))
                    ) : (
                      <div className="p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg text-yellow-500 text-sm">
                        🚧 该部位暂无训练数据
                      </div>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-4">
              <div className="w-16 h-16 rounded-full bg-gray-700/50 flex items-center justify-center">
                👆
              </div>
              <p className="text-center">请点击左侧模型</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}