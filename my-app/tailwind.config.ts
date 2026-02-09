import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // 这里把 CSS 变量映射为 Tailwind 类名
        background: "var(--background)",
        foreground: "var(--foreground)",
        
        // 卡片颜色
        card: {
          primary: "var(--card-primary)",     // 使用: bg-card-primary
          secondary: "var(--card-secondary)", // 使用: bg-card-secondary
          text: "var(--card-text)",           // 使用: text-card-text
        },

        searchbg: "var(--search-bg)",

        leftbox: "var(--left-box)",
        rightbox: "var(--right-box)",
        
      },
    },
  },
  plugins: [],
};
export default config;