'use client';

import { createContext, useContext, useEffect, useState } from 'react';

export type Theme = 'default' | 'light' | 'purple';

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('default');

  // 当 theme 改变时，修改 html 标签上的 data-theme 属性
  useEffect(() => {
    const root = document.documentElement;
    // 移除之前的属性
    root.removeAttribute('data-theme');
    
    if (theme !== 'default') {
      root.setAttribute('data-theme', theme);
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within a ThemeProvider');
  return context;
};