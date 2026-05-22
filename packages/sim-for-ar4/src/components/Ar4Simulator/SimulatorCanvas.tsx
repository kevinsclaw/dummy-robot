import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import { Suspense } from 'react';
import Ar4RobotModel from './Ar4RobotModel';
import { JointAngles } from './types';

interface SimulatorCanvasProps {
  jointAngles: JointAngles;
  targetAngles: JointAngles;
  onAnglesUpdate: (angles: JointAngles) => void;
}

export default function SimulatorCanvas({
  jointAngles,
  targetAngles,
  onAnglesUpdate,
}: SimulatorCanvasProps) {
  return (
    <Canvas
      shadows
      camera={{ position: [2.2, 1.6, 2.2], fov: 45, near: 0.1, far: 100 }}
      style={{ background: '#1a1a2e' }}
      aria-label="AR4 robot arm 3D simulator"
    >
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={20}
        shadow-camera-left={-5}
        shadow-camera-right={5}
        shadow-camera-top={5}
        shadow-camera-bottom={-5}
      />
      <pointLight position={[-3, 4, -3]} intensity={0.5} color="#4a90d9" />

      {/* Environment */}
      <Environment preset="city" />

      {/* Ground grid */}
      <Grid
        position={[0, 0, 0]}
        args={[10, 10]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#2a2a4a"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#4a4a8a"
        fadeDistance={8}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid
      />

      {/* Ground plane for shadows */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <shadowMaterial opacity={0.3} />
      </mesh>

      {/* AR4 Robot */}
      <Suspense fallback={null}>
        <Ar4RobotModel
          jointAngles={jointAngles}
          targetAngles={targetAngles}
          onAnglesUpdate={onAnglesUpdate}
        />
      </Suspense>

      {/* Camera controls */}
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={1}
        maxDistance={8}
        target={[0, 0.6, 0.4]}
      />
    </Canvas>
  );
}
