// app/data/muscles.ts

// 这是一个 TypeScript 接口，定义数据的格式
export interface MuscleData {
  name: string;      // 中文名 (显示在标签和侧边栏)
  actions: string[]; // 训练动作
  description?: string; // (可选) 肌肉描述
}

// 核心：翻译字典
// Key (键) = 模型里的怪名字 (如 Object_16)
// Value (值) = 你想要显示的数据
export const MUSCLE_MAP: Record<string, MuscleData> = {
  // === 胸部 ===
  'Object_16': { 
    name: '胸大肌', 
    actions: ['杠铃卧推', '哑铃飞鸟', '俯卧撑'] 
  },
  'Object_14': { // 假设这是腹肌
    name: '腹直肌', 
    actions: ['卷腹', '平板支撑'] 
  },
  'Object_10': { // 假设这是腹肌
    name: '三角肌', 
    actions: ['哑铃推举 (中束/前束)','侧平举 (中束)','反向飞鸟 (后束)'] 
  },
  'Object_11': { // 假设这是腹肌
    name: '上臂肌群', 
    actions: ['哑铃推举 (中束/前束)','侧平举 (中束)','反向飞鸟 (后束)'] 
  },
  'Object_22': { // 假设这是腹肌
    name: '前臂肌群', 
    actions: ['反握弯举','农夫行走'] 
  },
  'Object_20': { // 假设这是腹肌
    name: '股四头肌', 
    actions: ['深蹲','倒蹬 (腿举)','腿屈伸'] 
  },
  'Object_5': { // 假设这是腹肌
    name: '比目鱼肌', 
    actions: ['提踵 (站姿/坐姿)'] 
  },
  'Object_6': { // 假设这是腹肌
    name: '斜方肌', 
    actions: ['引体向上 (背阔)','高位下拉','划船 (厚度)','面拉 (斜方)'] 
  },
  'Object_8': { // 假设这是腹肌
    name: '下背', 
    actions: ['硬拉','山羊挺身'] 
  },
  'Object_18': { // 假设这是腹肌
    name: '臀大肌', 
    actions: ['臀桥/臀推', '宽距深蹲', '后撤箭步蹲', '器械髋外展'] 
  },
  
};

// 一个辅助函数：为了防止点到没数据的部位报错
export const getMuscleInfo = (meshName: string) => {
  return MUSCLE_MAP[meshName] || { 
    name: `未知部位 (${meshName})`, // 如果没配数据，就显示原始ID方便调试
    actions: [] 
  };
};