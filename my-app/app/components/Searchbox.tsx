// app/components/Searchbox.tsx
import { Search } from "lucide-react";

interface SearchboxProps {
  value: string;
  onChange: (value: string) => void;
}

export default function Searchbox({ value, onChange }: SearchboxProps) {
  return (
    <div className="z-10 w-full max-w-2xl px-6">
      <h2 className="text-2xl font-bold mb-4 text-center text-lime-400">
        FIND YOUR MOVE
      </h2>
      <div className="relative group">
        <input
          type="text"
          placeholder="搜索动作 (e.g. 深蹲, 卧推...)"
          // 1. 绑定父组件传来的 value
          value={value}
          // 2. 当输入改变时，调用父组件的方法
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-searchbg border border-zinc-700 rounded-full py-4 pl-12 pr-6 text-foreground focus:outline-none focus:border-lime-400 focus:ring-1 focus:ring-lime-400 transition-all placeholder:text-zinc-500"
        />
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400" />
      </div>
    </div>
  );
}