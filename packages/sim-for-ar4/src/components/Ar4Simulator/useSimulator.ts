import { useState, useCallback } from 'react';
import { JointAngles, Ros2Command, NAMED_POSES, DEFAULT_JOINT_ANGLES } from './types';
import { ChatMessage } from './InstructionPanel';

/**
 * Parse the ```ros2 [...] ``` fenced block from the agent's completed response.
 * Returns an empty array if no valid block is found.
 */
function extractRos2Commands(text: string): Ros2Command[] {
  const match = text.match(/```ros2\s*([\s\S]*?)```/);
  if (!match) return [];
  try {
    const parsed = JSON.parse(match[1].trim());
    const arr: Ros2Command[] = Array.isArray(parsed) ? parsed : [parsed];
    return arr.filter((c) => typeof c?.ros2_command === 'string');
  } catch {
    return [];
  }
}

/**
 * Apply a single ROS2 command to the current target joint angles.
 */
function applyCommand(current: JointAngles, command: Ros2Command): JointAngles {
  const next = { ...current };

  switch (command.ros2_command) {
    case 'joint_trajectory': {
      const joint = command.joint as keyof JointAngles | undefined;
      if (joint && joint in next && command.angle_degrees !== undefined) {
        next[joint] = command.angle_degrees;
      }
      break;
    }
    case 'named_pose': {
      if (command.pose_name && command.pose_name in NAMED_POSES) {
        return { ...NAMED_POSES[command.pose_name] };
      }
      if (command.joint_angles) {
        return { ...next, ...(command.joint_angles as Partial<JointAngles>) };
      }
      break;
    }
    case 'gripper': {
      if (command.action === 'open') {
        next.gripper = command.percentage ?? 100;
      } else if (command.action === 'close') {
        next.gripper = 0;
      }
      break;
    }
    case 'get_joint_states':
      break;
  }

  return next;
}

export interface SimulatorState {
  jointAngles: JointAngles;
  targetAngles: JointAngles;
  messages: ChatMessage[];
  isLoading: boolean;
  sendInstruction: (instruction: string) => Promise<void>;
  updateJointAngles: (angles: JointAngles) => void;
}

export function useSimulator(
  invokeAgent: (message: string, onChunk: (chunk: string) => void) => Promise<void>,
): SimulatorState {
  const [jointAngles, setJointAngles] = useState<JointAngles>({ ...DEFAULT_JOINT_ANGLES });
  const [targetAngles, setTargetAngles] = useState<JointAngles>({ ...DEFAULT_JOINT_ANGLES });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const updateJointAngles = useCallback((angles: JointAngles) => {
    setJointAngles(angles);
  }, []);

  const sendInstruction = useCallback(
    async (instruction: string) => {
      setMessages((prev) => [
        ...prev,
        { id: `user-${Date.now()}`, role: 'user', content: instruction, timestamp: new Date() },
      ]);
      setIsLoading(true);

      const assistantId = `assistant-${Date.now()}`;
      let fullText = '';

      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: 'assistant', content: '', commands: [], timestamp: new Date() },
      ]);

      try {
        await invokeAgent(instruction, (chunk: string) => {
          fullText += chunk;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: fullText } : m)),
          );
        });

        // Parse commands from the completed response and apply them all at once
        console.log('[Simulator] Full agent response:', fullText);
        const commands = extractRos2Commands(fullText);
        console.log('[Simulator] Parsed commands:', commands);

        if (commands.length > 0) {
          setTargetAngles((prev) => {
            let next = { ...prev };
            for (const cmd of commands) {
              next = applyCommand(next, cmd);
            }
            console.log('[Simulator] New targetAngles:', next);
            return next;
          });

          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, commands } : m)),
          );
        }
      } catch (err) {
        const errorText =
          err instanceof Error ? err.message : 'Failed to contact the agent. Please try again.';
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: `⚠️ Error: ${errorText}` } : m,
          ),
        );
      } finally {
        setIsLoading(false);
      }
    },
    [invokeAgent],
  );

  return { jointAngles, targetAngles, messages, isLoading, sendInstruction, updateJointAngles };
}
