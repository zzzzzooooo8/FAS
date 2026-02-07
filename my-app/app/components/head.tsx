// components/Header.tsx
import Link from 'next/link';

export default function Header() {
  return (
    <header className="
      fixed top-0 left-0 w-full z-50 
      bg-gradient-to-b from-gray-900/80 to-transparent
    ">
      
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        
        {/* 左边：首页 */}
        <Link href="/" className="group">
          <span className="text-2xl font-black text-[#E8FC5D] tracking-tighter group-hover:text-gray-300 transition-colors">
            HOME
          </span>
        </Link>

        {/* 右边：三个主题按钮 */}
        <div className="flex items-center gap-4">
          <button 
            className="w-6 h-6 rounded-full border-2 border-transparent hover:border-white hover:scale-110 transition-all cursor-pointer bg-[#E8FC5D]" 
            aria-label="Theme Black to Neon" 
          />

          <button 
            className="w-6 h-6 rounded-full border-2 border-transparent hover:scale-110 hover:border-white transition-all cursor-pointer bg-white" 
            aria-label="Theme White to Neon" 
          />

          <button 
            className="w-6 h-6 rounded-full border-2 border-transparent hover:scale-110 hover:border-white transition-all cursor-pointer bg-[#d0bbee]" 
            aria-label="Theme Purple to White" 
          />
          
        </div>
      </div>
    </header>
  );
}