"use client"; 

import Link from "next/link";
import ThemeButton from "@/app/components/Themebutton"

export default function Header() {

  return (
    <header
      className="
      fixed top-0 left-0 w-full z-50 
      bg-gradient-to-b from-black/20 to-transparent backdrop-blur-[2px]
    "
    >
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        {/* 左边：首页 - 动态文字颜色 */}
        <Link href="/" className="group">
          <span className="text-2xl font-black tracking-tighter text-[#E8FC5D] transition-colors duration-300 group-hover:opacity-70">
            HOME
          </span>
        </Link>

        {/* 右边：三个主题按钮 */}
        <div className="flex items-center gap-4">
          <ThemeButton bgcolor='#E8FC5D' targetTheme='default'/>
          <ThemeButton bgcolor='#F2F2E8' targetTheme='light'/>
          <ThemeButton bgcolor='#E4DAF6' targetTheme='purple'/>
        </div>
      </div>
    </header>
  );
}
