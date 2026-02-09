import WorkoutDashboard from "@/app/components/WorkoutDashboard"
import { EXERCISES_DATA } from "@/app/data/exercises"

export default function Page() {
  // 在这里模拟服务端数据获取
  // 以后如果你有了真实数据库，可以在这里写: await db.query(...)
  const exercises = EXERCISES_DATA;

  return (
    <div className="h-[100dvh]  w-full flex flex-col bg-background text-white overflow-hidden">
      
      {/* 头部占位块 (Server Side Rendered) */}
      <div className="h-20 w-full shrink-0" />

      {/* 加载交互组件，并将数据作为 Props 传入 */}
      {/* 所有的 useState 和 onClick 逻辑都封装在这个组件内部 */}
      <WorkoutDashboard initialExercises={exercises} />
      
    </div>
  )
}