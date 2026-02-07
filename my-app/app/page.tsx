import FitnessCard from "@/app/components/selectCard";

export default function Home() {
  return (
    <>
      {/*背景 */}
      <div className="bg-slate-950 w-full min-h-screen flex flex-col items-center">
        {/*招呼文字 */}
        <div className="h-[50vh] flex flex-col items-center justify-center">
          <span className="text-white text-2xl font-light tracking-widest uppercase mb-2">
            Welcome to
          </span>
          <h1 className="text-white text-8xl font-black tracking-tighter">
            ZOOLIN FITNESS
          </h1>
        </div>

        {/*选择卡片 */}
        <div className="flex gap-20">
            <FitnessCard
              title="WORKOUT"
              description="挑选心仪的练习动作，即刻开启身材蜕变！"
              href="/workout"
              bgClass="bg-[#E8FC5D]" // 自定义红色
              textClass="text-black"
              rotateClass="-rotate-2" // 向左倾斜 2度
            />

            {/* 卡片 2：亮白色系，向右歪一点 */}
            <FitnessCard
              title="DIFFERENT PARTS"
              description="聚焦你想提升的区域，雕刻完美身体线条！"
              href="/part"
              bgClass="bg-[#F7F8F3]" // 亮白色背景
              textClass="text-slate-900" // 深色文字
              rotateClass="rotate-3" // 向右倾斜 3度
            />
        </div>
      </div>
    </>
  );
}
