import MuscleTrainingFeature from '@/app/components/MuscleTrainingFeature';

export const metadata = {
  title: '3D 健身模型展示 | Zoolin Fitness',
  description: '交互式3D人体肌肉解剖与训练指南',
};

// 这是一个标准的 Server Component
export default function TrainingPage() {
  // 服务端组件可以直接进行数据库查询 (比如 fetch 用户权限等)，如果有需要的话
  // await checkUserSession(); 

  return (
    <main className="flex w-full h-screen bg-black">
      {/* 这里引入了 Client Component。
        Next.js 会在服务端预渲染它的 HTML 结构，
        然后在客户端 "Hydrate" (注水) 使其可交互。
      */}
      <MuscleTrainingFeature />
    </main>
  );
}