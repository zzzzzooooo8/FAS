import {Search} from "lucide-react";

export default function WorkoutPage() {
  // 定义目录数据，方便复用
  const categories = ["臀部", "背部", "胳膊", "腿部", "胸部", "腹部"];

  return (
    // pt-20 是为了避开你那个 fixed 的 Header
    <div className="h-screen w-full flex flex-col pt-20 bg-black text-white overflow-hidden">
      
      {/* ==================== 
          1. 顶部 1/3：搜索区域 
      ==================== */}
      <div className="h-[33vh] w-full flex flex-col items-center justify-center bg-zinc-900/50 border-b border-white/10 relative">
         {/* 装饰性背景标题 */}
         <h1 className="absolute text-8xl font-black text-white/5 select-none pointer-events-none tracking-tighter">
            SEARCH
         </h1>

         {/* 搜索框主体 */}
         <div className="z-10 w-full max-w-2xl px-6">
            <h2 className="text-2xl font-bold mb-4 text-center text-lime-400">
               FIND YOUR MOVE
            </h2>
            <div className="relative group">
                <input 
                  type="text" 
                  placeholder="搜索动作 (e.g. 深蹲, 卧推...)"
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-full py-4 pl-12 pr-6 text-white focus:outline-none focus:border-lime-400 focus:ring-1 focus:ring-lime-400 transition-all placeholder:text-zinc-500"
                />
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400" />
            </div>
         </div>
      </div>

      {/* ==================== 
          2. 剩余部分：左右分栏 
          flex-1 会自动填满剩下的 2/3 高度
      ==================== */}
      <div className="flex-1 grid grid-cols-2">
        
        {/* === 左边框：无器械 (目录在左) === */}
        {/* bg-zinc-950: 深色背景 */}
        <div className="flex h-full border-r border-white/10 bg-zinc-950">
           
           {/* 左侧目录栏 */}
           <div className="w-24 h-full border-r border-white/5 flex flex-col py-6 bg-zinc-900/50">
              {categories.map((item) => (
                <button key={item} className="py-4 text-zinc-400 hover:text-lime-400 hover:bg-zinc-800 transition-all font-bold text-sm tracking-widest text-center border-l-2 border-transparent hover:border-lime-400">
                  {item}
                </button>
              ))}
           </div>

           {/* 左侧内容展示区 */}
           <div className="flex-1 p-8 flex flex-col items-center justify-center relative overflow-hidden group">
              <h3 className="text-3xl font-black italic text-zinc-700 group-hover:text-zinc-600 transition-colors uppercase mb-2">
                 Bodyweight
              </h3>
              <p className="text-lime-400 font-medium">无器械训练</p>
              {/* 这里之后可以放具体的动作列表 */}
           </div>
        </div>

        {/* === 右边框：器械 (目录在右) === */}
        {/* bg-zinc-900: 稍微亮一点的背景，区分左右 */}
        <div className="flex h-full bg-zinc-900">
           
           {/* 右侧内容展示区 (flex-1 占满剩余空间) */}
           <div className="flex-1 p-8 flex flex-col items-center justify-center relative overflow-hidden group">
              <h3 className="text-3xl font-black italic text-zinc-700 group-hover:text-zinc-600 transition-colors uppercase mb-2">
                 Equipment
              </h3>
              <p className="text-blue-400 font-medium">器械训练</p>
           </div>

           {/* 右侧目录栏 (放在 flex 容器的最后，自然就在最右边了) */}
           <div className="w-24 h-full border-l border-white/5 flex flex-col py-6 bg-zinc-800/50">
              {categories.map((item) => (
                <button key={item} className="py-4 text-zinc-400 hover:text-blue-400 hover:bg-zinc-700 transition-all font-bold text-sm tracking-widest text-center border-r-2 border-transparent hover:border-blue-400">
                  {item}
                </button>
              ))}
           </div>

        </div>

      </div>
    </div>
  )
}