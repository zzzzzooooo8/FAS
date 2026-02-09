'use client';

import { useTheme, Theme } from '@/app/context/ThemeContext';

interface ThemeButtonProps {
  bgcolor: string;      
  targetTheme: Theme;  
}

export default function ThemeButton({ bgcolor, targetTheme }: ThemeButtonProps) {
  const { theme, setTheme } = useTheme();

  // 判断当前按钮是否是激活状态
  const isActive = theme === targetTheme;

  return (
    <button
      onClick={() => setTheme(targetTheme)} // 点击时切换到目标主题
      style={{ 
        backgroundColor: bgcolor,
        boxShadow: isActive ? `0 0 10px ${bgcolor}` : 'none'
      }}
      className={`
        w-6 h-6 rounded-full border-2 transition-all cursor-pointer
        ${isActive
            ? "border-white scale-125" 
            : "border-transparent opacity-60 hover:opacity-100 hover:scale-110" 
        }
      `}
      aria-label={`Switch to ${targetTheme} theme`}
    />
  );
}