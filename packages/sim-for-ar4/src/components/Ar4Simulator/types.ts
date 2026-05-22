/**
 * Joint angles for the AR4 6-DOF robot arm (all in degrees).
 */
export interface JointAngles {
  base: number; // ±180°
  shoulder: number; // -90° to 90°
  elbow: number; // -135° to 135°
  wrist_pitch: number; // -90° to 90°
  wrist_roll: number; // ±180°
  gripper: number; // 0 (closed) to 100 (open %)
}

export const DEFAULT_JOINT_ANGLES: JointAngles = {
  base: 0,
  shoulder: 0,
  elbow: 0,
  wrist_pitch: 0,
  wrist_roll: 0,
  gripper: 0,
};

/**
 * A ROS2 command emitted by the Strands agent.
 */
export interface Ros2Command {
  ros2_command: 'joint_trajectory' | 'named_pose' | 'gripper' | 'get_joint_states';
  joint?: string;
  angle_degrees?: number;
  speed?: number;
  pose_name?: string;
  joint_angles?: Partial<JointAngles>;
  action?: 'open' | 'close';
  percentage?: number;
  force?: number;
  topic?: string;
}

/**
 * Named poses for the AR4 robot.
 */
export const NAMED_POSES: Record<string, JointAngles> = {
  home: { base: 0, shoulder: 0, elbow: 0, wrist_pitch: 0, wrist_roll: 0, gripper: 0 },
  ready: { base: 0, shoulder: -30, elbow: 60, wrist_pitch: -30, wrist_roll: 0, gripper: 0 },
  pick_up: { base: 0, shoulder: -60, elbow: 90, wrist_pitch: -30, wrist_roll: 0, gripper: 100 },
  place_down: { base: 90, shoulder: -45, elbow: 75, wrist_pitch: -30, wrist_roll: 0, gripper: 0 },
  wave: { base: 45, shoulder: -45, elbow: 90, wrist_pitch: 45, wrist_roll: 0, gripper: 0 },
  inspect: { base: 0, shoulder: -15, elbow: 30, wrist_pitch: -15, wrist_roll: 0, gripper: 50 },
  rest: { base: 0, shoulder: 0, elbow: 0, wrist_pitch: 0, wrist_roll: 0, gripper: 0 },
};
