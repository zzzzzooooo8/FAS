import Link from 'next/link';

interface CardProps {
  title: string;        // 标题
  description: string;  // 描述
  href: string;         // 跳转链接
  bgClass?: string;     // 背景颜色类名 
  textClass?: string;   // 文字颜色类名 
  rotateClass?: string; // 倾斜角度类名 
}

const FitnessCard = ({ 
  title, 
  description, 
  href, 
  // 是默认值
  bgClass = "bg-zinc-800", 
  textClass = "text-white",
  rotateClass = "rotate-0" 
}: CardProps) => {
  return (
    <Link href={href} className="group block">
      <div className={`
        ${bgClass} ${textClass} ${rotateClass}
        w-full p-16 py-20 rounded-3xl
        transform transition-all duration-300 ease-out
        hover:scale-110 hover:shadow-2xl hover:rotate-0
        border border-white/10
      `}>
        <h3 className="text-3xl font-black mb-2 uppercase italic">
          {title}
        </h3>
        <p className="opacity-80 font-medium">
          {description}
        </p>
        
        {/* 小箭头图标*/}
        <div className="mt-6 flex justify-end">
           <span className="text-2xl group-hover:translate-x-2 transition-transform">→</span>
        </div>
      </div>
    </Link>
  );
};

export default FitnessCard;