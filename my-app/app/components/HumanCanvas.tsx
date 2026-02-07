'use client';
import { getMuscleInfo } from '@/app/data/muscles';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Canvas, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, useGLTF, ContactShadows, Html } from '@react-three/drei';
import * as THREE from 'three';

// === 定义父组件传进来的 Props ===
// 这样你的 Page 页面才能和这个 Canvas 通信
interface HumanCanvasProps {
  onMuscleSelect?: (muscleName: string) => void; // 告诉父组件谁被点了
}

// === 定义高亮材质 (你可以改成任何醒目的颜色) ===
// emissive 属性让它看起来发光，color 是基础色
const HIGHLIGHT_MATERIAL = new THREE.MeshStandardMaterial({
  color: '#ff0055',    // 醒目的洋红色
  emissive: '#550022', // 自发光微红
  roughness: 0.2,
  metalness: 0.8
});

function HumanModel({ onMuscleSelect }: HumanCanvasProps) {
  // 加载模型
  const { scene } = useGLTF('/model/human.glb');
  
  // 1. 本地状态：记录当前选中的肌肉名字，以及标签显示的位置
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [labelPosition, setLabelPosition] = useState<[number, number, number]>([0, 0, 0]);

  // 2. 存储原始材质的“备份库”
  // key是Mesh的名字，value是它原本的材质。用于取消选中时恢复原样。
  const originalMaterials = useRef<Record<string, THREE.Material | THREE.Material[]>>({});

  // 3. 克隆场景 (标准操作，防止缓存污染)
  const sceneClone = useMemo(() => scene.clone(), [scene]);

  // === 核心逻辑：监听点击 ===
  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation(); // 防止点击穿透到地板

    // e.object 就是你点击到的那个具体的“肌肉” Mesh
    const mesh = e.object as THREE.Mesh;
    const meshName = mesh.name;

    // 如果点的不是 Mesh (可能是骨骼或其他)，忽略
    if (!mesh.isMesh) return;

    // 更新位置：e.point 是你点击模型表面的 3D 坐标
    setLabelPosition([e.point.x, e.point.y + 0.2, e.point.z]); // y+0.2 让标签稍微浮起来一点

    // 更新选中状态
    setSelectedName(meshName);
    
    // 告诉父组件 (Page页面)
    if (onMuscleSelect) {
      onMuscleSelect(meshName);
    }
  };

  // === 核心逻辑：材质替换 ===
  // 每当 selectedName 变化时，执行这个副作用
  useEffect(() => {
    // 遍历整个模型
    sceneClone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;

        // 第一次遍历到这个 Mesh 时，把它的原始材质存进备份库
        if (!originalMaterials.current[mesh.name]) {
          originalMaterials.current[mesh.name] = mesh.material;
        }

        // 判断：我是被选中的那个吗？
        if (mesh.name === selectedName) {
          // 是 -> 换成高亮材质
          mesh.material = HIGHLIGHT_MATERIAL;
        } else {
          // 否 -> 恢复成备份库里的原始材质
          if (originalMaterials.current[mesh.name]) {
            mesh.material = originalMaterials.current[mesh.name];
          }
        }
      }
    });
  }, [selectedName, sceneClone]); // 依赖项：当选中名字变了，重新刷一遍材质

  return (
    <group>
      <primitive 
        object={sceneClone} 
        scale={2.5} 
        position={[0, 0, 0]} // 这里调了一下Y轴，通常人脚底要在0点，需要往下移一点
        onPointerDown={handlePointerDown} // 【关键】在这里绑定点击事件
      />
      
      {/* === 悬浮标签 === */}
      {selectedName && (
        <Html position={labelPosition} center>
          <div className="pointer-events-none px-3 py-1 bg-black/80 text-white text-sm rounded-full backdrop-blur-sm border border-white/20 shadow-xl animate-in fade-in zoom-in duration-200">
             {/* 这里显示肌肉名，你可以加一个字典把英文 mesh name 转成中文 */}
             {getMuscleInfo(selectedName).name}
          </div>
        </Html>
      )}
    </group>
  );
}

// === 主画布组件 ===
// 这里的 props 也要透传下去
export default function HumanCanvas({ onMuscleSelect }: HumanCanvasProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        shadows
        camera={{ position: [0, 0, 6], fov: 50 }} 
        className="bg-transparent" 
      >
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4f46e5" />

        <React.Suspense fallback={null}>
            {/* 把父组件传来的函数传给模型 */}
            <HumanModel onMuscleSelect={onMuscleSelect} />  
        </React.Suspense>

        <OrbitControls 
            enablePan={false} 
            minPolarAngle={Math.PI / 4} 
            maxPolarAngle={Math.PI / 1.6} 
            minDistance={3} 
            maxDistance={8} 
        />
        
        <ContactShadows resolution={1024} scale={10} blur={1} opacity={0.5} far={10} color="#000000" />
      </Canvas>
    </div>
  );
}