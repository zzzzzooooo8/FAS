// app/train/[action]/page.tsx
import { notFound } from "next/navigation";
import { EXERCISES_DATA } from "@/app/data/exercises";
import TrainingView from "@/app/components/TrainingView"; // 确保路径对

interface PageProps {
  params: Promise<{
    action: string;
  }>;
}

export default async function TrainingPage({ params }: PageProps) {
  // Next.js 15: params 必须 await
  const resolvedParams = await params;
  const actionSlug = resolvedParams.action;

  // 1. 在服务端找数据
  const exercise = EXERCISES_DATA.find((e) => {
    // 容错处理：转成字符串并去空格
    return String(e.idName).trim() === String(actionSlug).trim();
  });

  // 2. 找不到就 404
  if (!exercise) {
    notFound(); 
  }

  // 3. 找到后，传给客户端组件
  return <TrainingView exercise={exercise} />;
}