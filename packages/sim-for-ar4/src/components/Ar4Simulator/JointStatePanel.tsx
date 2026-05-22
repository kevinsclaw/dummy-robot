import { JointAngles } from './types';

interface JointStatePanelProps {
  jointAngles: JointAngles;
  targetAngles: JointAngles;
}

interface JointConfig {
  key: keyof JointAngles;
  label: string;
  min: number;
  max: number;
  unit: string;
}

const JOINT_CONFIGS: JointConfig[] = [
  { key: 'base', label: 'Base', min: -180, max: 180, unit: '°' },
  { key: 'shoulder', label: 'Shoulder', min: -90, max: 90, unit: '°' },
  { key: 'elbow', label: 'Elbow', min: -135, max: 135, unit: '°' },
  { key: 'wrist_pitch', label: 'Wrist Pitch', min: -90, max: 90, unit: '°' },
  { key: 'wrist_roll', label: 'Wrist Roll', min: -180, max: 180, unit: '°' },
  { key: 'gripper', label: 'Gripper', min: 0, max: 100, unit: '%' },
];

function JointBar({ value, min, max }: { value: number; min: number; max: number }) {
  const range = max - min;
  const pct = ((value - min) / range) * 100;
  const isMoving = false;

  return (
    <div
      className="relative h-2 rounded-full overflow-hidden"
      style={{ background: 'rgba(255,255,255,0.1)' }}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={min}
      aria-valuemax={max}
    >
      <div
        className="absolute top-0 left-0 h-full rounded-full transition-all duration-100"
        style={{
          width: `${pct}%`,
          background: isMoving
            ? 'linear-gradient(90deg, #e67e22, #f39c12)'
            : 'linear-gradient(90deg, #4a90d9, #27ae60)',
        }}
      />
    </div>
  );
}

export default function JointStatePanel({ jointAngles, targetAngles }: JointStatePanelProps) {
  return (
    <div
      className="rounded-lg p-4 space-y-3"
      style={{ background: 'rgba(0,0,0,0.6)', border: '1px solid rgba(74,144,217,0.3)' }}
    >
      <h3 className="text-sm font-semibold text-blue-300 uppercase tracking-wider">
        Joint States
      </h3>
      {JOINT_CONFIGS.map(({ key, label, min, max, unit }) => {
        const current = jointAngles[key];
        const target = targetAngles[key];
        const isMoving = Math.abs(current - target) > 0.5;

        return (
          <div key={key} className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-300">{label}</span>
              <div className="flex items-center gap-2">
                {isMoving && (
                  <span className="text-xs text-orange-400 animate-pulse">→ {target.toFixed(1)}{unit}</span>
                )}
                <span
                  className="text-xs font-mono font-semibold"
                  style={{ color: isMoving ? '#e67e22' : '#4a90d9' }}
                >
                  {current.toFixed(1)}{unit}
                </span>
              </div>
            </div>
            <JointBar value={current} min={min} max={max} />
          </div>
        );
      })}
    </div>
  );
}
