import { useRef, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { JointAngles } from './types';

interface Ar4RobotModelProps {
  jointAngles: JointAngles;
  targetAngles: JointAngles;
  onAnglesUpdate: (angles: JointAngles) => void;
}

const DEG = Math.PI / 180;
const LERP_SPEED = 0.08;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

// Segment lengths
const SHOULDER_H = 0.42;
const UPPER_LEN  = 0.72;
const FORE_LEN   = 0.62;
const WRIST_LEN  = 0.28;

export default function Ar4RobotModel({
  jointAngles,
  targetAngles,
  onAnglesUpdate,
}: Ar4RobotModelProps) {
  // ── Joint pivot refs ──────────────────────────────────────────────────────
  const baseRef        = useRef<THREE.Group>(null);
  const shoulderRef    = useRef<THREE.Group>(null);
  const elbowRef       = useRef<THREE.Group>(null);
  const wristPitchRef  = useRef<THREE.Group>(null);
  const wristRollRef   = useRef<THREE.Group>(null);
  const gripperLeftRef = useRef<THREE.Group>(null);
  const gripperRightRef= useRef<THREE.Group>(null);

  // ── Mutable refs so useFrame always sees the latest prop values ───────────
  // useFrame's callback is registered once; it closes over stale props unless
  // we forward updates through refs.
  const jointRef  = useRef<JointAngles>(jointAngles);
  const targetRef = useRef<JointAngles>(targetAngles);
  const cbRef     = useRef<(angles: JointAngles) => void>(onAnglesUpdate);

  useEffect(() => { jointRef.current  = jointAngles;   }, [jointAngles]);
  useEffect(() => { targetRef.current = targetAngles;  }, [targetAngles]);
  useEffect(() => { cbRef.current     = onAnglesUpdate; }, [onAnglesUpdate]);

  const frameCount = useRef(0);

  // ── Animation loop ────────────────────────────────────────────────────────
  useFrame(() => {
    frameCount.current++;
    // Log once every 120 frames so we can see live values without flooding
    const shouldLog = frameCount.current % 120 === 1;

    const cur    = { ...jointRef.current };
    const target = targetRef.current;

    if (shouldLog) {
      console.log('[useFrame] cur:', JSON.stringify(cur), '| target:', JSON.stringify(target));
    }

    let changed  = false;

    for (const key of Object.keys(cur) as (keyof JointAngles)[]) {
      const next = lerp(cur[key], target[key], LERP_SPEED);
      if (Math.abs(next - target[key]) > 0.01) {
        cur[key] = next;
        changed  = true;
      } else {
        cur[key] = target[key];
      }
    }

    // Write back so next frame starts from the interpolated position
    jointRef.current = cur;
    if (changed) cbRef.current(cur);

    // Apply to Three.js objects directly — no React re-render needed
    if (baseRef.current)        baseRef.current.rotation.y        = cur.base        * DEG;
    if (shoulderRef.current)    shoulderRef.current.rotation.x    = cur.shoulder    * DEG;
    if (elbowRef.current)       elbowRef.current.rotation.x       = cur.elbow       * DEG;
    if (wristPitchRef.current)  wristPitchRef.current.rotation.x  = cur.wrist_pitch * DEG;
    if (wristRollRef.current)   wristRollRef.current.rotation.z   = cur.wrist_roll  * DEG;

    const spread = (cur.gripper / 100) * 0.10;
    if (gripperLeftRef.current)  gripperLeftRef.current.position.x  = -(0.03 + spread);
    if (gripperRightRef.current) gripperRightRef.current.position.x =  (0.03 + spread);
  });

  // ── Geometry ──────────────────────────────────────────────────────────────
  return (
    <group>
      {/* Base plate */}
      <mesh position={[0, 0.04, 0]} receiveShadow castShadow>
        <cylinderGeometry args={[0.32, 0.38, 0.08, 32]} />
        <meshStandardMaterial color="#1e2a3a" metalness={0.9} roughness={0.2} />
      </mesh>

      {/* Static base column */}
      <mesh position={[0, 0.22, 0]} castShadow>
        <cylinderGeometry args={[0.10, 0.13, 0.28, 24]} />
        <meshStandardMaterial color="#2e4a6a" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* J1 — Base rotation (Y) */}
      <group ref={baseRef} position={[0, 0.08, 0]}>

        <mesh position={[0, SHOULDER_H - 0.06, 0]} castShadow>
          <boxGeometry args={[0.18, 0.12, 0.18]} />
          <meshStandardMaterial color="#2e4a6a" metalness={0.8} roughness={0.3} />
        </mesh>

        <mesh position={[0, SHOULDER_H, 0]} castShadow>
          <cylinderGeometry args={[0.10, 0.10, 0.06, 24]} />
          <meshStandardMaterial color="#e67e22" metalness={0.6} roughness={0.4} />
        </mesh>

        {/* J2 — Shoulder (X) */}
        <group ref={shoulderRef} position={[0, SHOULDER_H, 0]}>

          {/* Upper arm extends along +Z */}
          <mesh position={[0, 0, UPPER_LEN / 2]} castShadow>
            <boxGeometry args={[0.10, 0.10, UPPER_LEN]} />
            <meshStandardMaterial color="#4a90d9" metalness={0.7} roughness={0.3} />
          </mesh>

          <mesh position={[0, 0, UPPER_LEN]} castShadow>
            <sphereGeometry args={[0.09, 24, 24]} />
            <meshStandardMaterial color="#e67e22" metalness={0.6} roughness={0.4} />
          </mesh>

          {/* J3 — Elbow (X) */}
          <group ref={elbowRef} position={[0, 0, UPPER_LEN]}>

            {/* Forearm extends along +Z */}
            <mesh position={[0, 0, FORE_LEN / 2]} castShadow>
              <boxGeometry args={[0.09, 0.09, FORE_LEN]} />
              <meshStandardMaterial color="#4a90d9" metalness={0.7} roughness={0.3} />
            </mesh>

            <mesh position={[0, 0, FORE_LEN]} castShadow>
              <sphereGeometry args={[0.08, 24, 24]} />
              <meshStandardMaterial color="#e67e22" metalness={0.6} roughness={0.4} />
            </mesh>

            {/* J4 — Wrist pitch (X) */}
            <group ref={wristPitchRef} position={[0, 0, FORE_LEN]}>

              {/* Wrist link extends along +Z */}
              <mesh position={[0, 0, WRIST_LEN / 2]} castShadow>
                <boxGeometry args={[0.08, 0.08, WRIST_LEN]} />
                <meshStandardMaterial color="#3a7abf" metalness={0.7} roughness={0.3} />
              </mesh>

              <mesh position={[0, 0, WRIST_LEN]} rotation={[Math.PI / 2, 0, 0]} castShadow>
                <cylinderGeometry args={[0.07, 0.07, 0.06, 20]} />
                <meshStandardMaterial color="#1e2a3a" metalness={0.9} roughness={0.2} />
              </mesh>

              {/* J5 — Wrist roll (Z) */}
              <group ref={wristRollRef} position={[0, 0, WRIST_LEN]}>

                <mesh position={[0, 0, 0.06]} castShadow>
                  <boxGeometry args={[0.16, 0.07, 0.08]} />
                  <meshStandardMaterial color="#1e2a3a" metalness={0.9} roughness={0.2} />
                </mesh>

                {/* J6 — Gripper left finger */}
                <group ref={gripperLeftRef} position={[-0.03, 0, 0.06]}>
                  <mesh position={[0, 0, 0.10]} castShadow>
                    <boxGeometry args={[0.035, 0.055, 0.18]} />
                    <meshStandardMaterial color="#27ae60" metalness={0.5} roughness={0.5} />
                  </mesh>
                  <mesh position={[0, -0.01, 0.20]} castShadow>
                    <boxGeometry args={[0.035, 0.035, 0.04]} />
                    <meshStandardMaterial color="#1e8449" metalness={0.5} roughness={0.5} />
                  </mesh>
                </group>

                {/* J6 — Gripper right finger */}
                <group ref={gripperRightRef} position={[0.03, 0, 0.06]}>
                  <mesh position={[0, 0, 0.10]} castShadow>
                    <boxGeometry args={[0.035, 0.055, 0.18]} />
                    <meshStandardMaterial color="#27ae60" metalness={0.5} roughness={0.5} />
                  </mesh>
                  <mesh position={[0, -0.01, 0.20]} castShadow>
                    <boxGeometry args={[0.035, 0.035, 0.04]} />
                    <meshStandardMaterial color="#1e8449" metalness={0.5} roughness={0.5} />
                  </mesh>
                </group>

              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}
