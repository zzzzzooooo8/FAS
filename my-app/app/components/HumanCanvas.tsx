'use client';

import React, { useState, useEffect, useMemo, useRef, Suspense } from 'react';
import { Canvas, ThreeEvent } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, ContactShadows, Html } from '@react-three/drei';
import * as THREE from 'three';
// 假设你的数据文件在这里，如果路径不对请修改
import { getMuscleInfo } from '@/app/data/muscles'; 

// === 定义父组件传进来的 Props ===
interface HumanCanvasProps {
  onMuscleSelect?: (muscleName: string) => void;
}

// === 定义高亮材质 ===
// 当选中某个部位时，用这个材质覆盖它
const HIGHLIGHT_MATERIAL = new THREE.MeshStandardMaterial({
  color: '#ff0055',    // 醒目的洋红色
  emissive: '#550022', // 自发光微红
  roughness: 0.2,
  metalness: 0.8
});

function HumanModel({ onMuscleSelect }: HumanCanvasProps) {
  // 加载模型 (请确认 public/model/human.glb 存在)
  const { scene } = useGLTF('/model/human.glb');
  
  // 1. 本地状态
  const [selectedName, setSelectedName] = useState<string | null>(null);
  
  // 2. 存储原始材质的“备份库”
  // key是Mesh的名字，value是它原本的材质
  const originalMaterials = useRef<Record<string, THREE.Material | THREE.Material[]>>({});

  // 3. 克隆场景 (标准操作，防止缓存污染)
  const sceneClone = useMemo(() => scene.clone(), [scene]);

  // === 初始化：预处理模型材质，让它看起来不那么像塑料 ===
  useEffect(() => {
    sceneClone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // 备份原始材质
        if (!originalMaterials.current[mesh.name]) {
          originalMaterials.current[mesh.name] = mesh.material;
        }

        // 增强材质质感 (解决灰蒙蒙问题的一环)
        if (mesh.material instanceof THREE.MeshStandardMaterial) {
          mesh.material.envMapIntensity = 1.2; // 增强环境反射
          mesh.material.needsUpdate = true;
        }
      }
    });
  }, [sceneClone]);

  // === 核心逻辑：监听点击 ===
  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation(); // 防止点击穿透

    const mesh = e.object as THREE.Mesh;
    // 如果点的不是 Mesh，忽略
    if (!mesh.isMesh) return;

    const meshName = mesh.name;
    console.log("Clicked Mesh Name:", meshName); // 调试用：看看点了啥

    // 更新选中状态
    setSelectedName(meshName);
    
    // 告诉父组件
    if (onMuscleSelect) {
      onMuscleSelect(meshName);
    }
  };

  // === 核心逻辑：材质替换 (高亮选中项) ===
  useEffect(() => {
    sceneClone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        
        // 确保备份库里有它
        if (!originalMaterials.current[mesh.name]) {
             originalMaterials.current[mesh.name] = mesh.material;
        }

        if (mesh.name === selectedName) {
          // 选中 -> 换成高亮材质
          mesh.material = HIGHLIGHT_MATERIAL;
        } else {
          // 未选中 -> 恢复原始材质
          if (originalMaterials.current[mesh.name]) {
            mesh.material = originalMaterials.current[mesh.name];
          }
        }
      }
    });
  }, [selectedName, sceneClone]);

  return (
    <group>
      <primitive 
        object={sceneClone} 
        scale={2.5} 
        position={[0, 0, 0]} // 下移一点，让人站在画面中心
        onPointerDown={handlePointerDown} 
      />
    </group>
  );
}

// === 主画布组件 ===
export default function HumanCanvas({ onMuscleSelect }: HumanCanvasProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        shadows
        camera={{ position: [0, 0, 6], fov: 50 }} 
        className="bg-transparent" 
        dpr={[1, 2]} // 适配高分屏，更清晰
      >
        {/* 1. 环境光：基础亮度 */}
        <ambientLight intensity={0.5} />
        
        {/* 2. 主光源：模拟太阳光，产生阴影 */}
        <directionalLight 
          position={[5, 10, 7]} 
          intensity={1.5} 
          castShadow 
          shadow-mapSize={[1024, 1024]} 
        />
        
        {/* 3. 补光灯：照亮侧面 */}
        <spotLight position={[-10, 0, -5]} intensity={0.8} color="#ffffff" />

        {/* 4. ⭐ 核心：环境贴图 (让模型不再灰蒙蒙) */}
        <Environment preset="city" />

        <Suspense fallback={<Html center>Loading 3D Model...</Html>}>
           <HumanModel onMuscleSelect={onMuscleSelect} />  
        </Suspense>

        <OrbitControls 
           enablePan={false} 
           minPolarAngle={Math.PI / 2.5} 
           maxPolarAngle={Math.PI / 1.8}
           minDistance={3} 
           maxDistance={8} 
        />
        
        {/* 地面接触阴影 */}
        <ContactShadows position={[0, -2.5, 0]} opacity={0.4} scale={10} blur={2.5} far={4} />
      </Canvas>
    </div>
  );
}