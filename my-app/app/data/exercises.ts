// app/data/exercises.ts

// 1. 定义 TypeScript 接口，规范数据格式
export interface Exercise {
  id: number;
  idName: string;
  name: string;
  category: string;
  type: "bodyweight" | "equipment"; // 限制只能是这两个字符串
  difficulty: "L1" | "L2" | "L3";
  target?: string;
  image?: string;
}

// 2. 导出数据
export const EXERCISES_DATA: Exercise[] = [
  // === 臀部 (Glutes) ===
  {
    id: 1,
    idName: "squat",
    name: "徒手深蹲",
    category: "臀部",
    type: "bodyweight",
    difficulty: "L1",
    image: "/images/squat.png",
  },
  {
    id: 2,
    idName: "glute-bridge",
    name: "臀桥",
    category: "臀部",
    type: "bodyweight",
    difficulty: "L1",
    image: "/images/squat.png",
  },
  {
    id: 3,
    idName: "deadlift",
    name: "杠铃硬拉",
    category: "臀部",
    type: "equipment",
    difficulty: "L3",
    image: "/images/squat.png",
  },
  
  // === 背部 (Back) ===
  {
    id: 4,
    idName: "Back-Extension",
    name: "俯卧挺身",
    category: "背部",
    type: "bodyweight",
    difficulty: "L1",
    image: "/images/squat.png",
  },
  {
    id: 5,
    idName: "Lat-Pulldown",
    name: "高位下拉",
    category: "背部",
    type: "equipment",
    difficulty: "L2",
    image: "/images/squat.png",
  },

  // === 腿部 (Legs) ===
  {
    id: 6,
    idName: "Bulgarian-Split-Squat",
    name: "保加利亚蹲",
    category: "腿部",
    type: "bodyweight",
    difficulty: "L2",
    image: "/images/squat.png",
  },
  {
    id: 7,
    idName: "Leg-Press",
    name: "倒蹬机",
    category: "腿部",
    type: "equipment",
    difficulty: "L2",
    image: "/images/squat.png",
  },
  
  // ... 添加更多数据
];