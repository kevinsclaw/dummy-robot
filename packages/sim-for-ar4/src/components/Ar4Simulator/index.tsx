import { useCallback } from 'react';
import { useAgentClient } from '../../hooks/useAgentClient';
import SimulatorCanvas from './SimulatorCanvas';
import JointStatePanel from './JointStatePanel';
import InstructionPanel from './InstructionPanel';
import { useSimulator } from './useSimulator';

/**
 * AR4 Robot Simulator — main container component.
 *
 * Wires together:
 *  - 3D robot canvas (Three.js / React Three Fiber)
 *  - Joint state panel
 *  - Instruction input + Strands agent chat
 */
export default function Ar4Simulator() {
  const agentClient = useAgentClient();

  /**
   * Invoke the Strands agent with a streaming response using the generated client.
   * Calls onChunk for each text chunk received.
   */
  const invokeAgent = useCallback(
    async (message: string, onChunk: (chunk: string) => void): Promise<void> => {
      const stream = agentClient.invoke({ message });
      for await (const chunk of stream) {
        if (chunk.content) {
          onChunk(chunk.content);
        }
      }
    },
    [agentClient],
  );

  const { jointAngles, targetAngles, messages, isLoading, sendInstruction, updateJointAngles } =
    useSimulator(invokeAgent);

  return (
    <div
      className="flex h-full w-full overflow-hidden"
      style={{ background: '#0f0f1a', minHeight: '600px' }}
    >
      {/* Left panel: joint states */}
      <div
        className="flex flex-col gap-3 p-3 overflow-y-auto"
        style={{ width: '220px', minWidth: '220px', borderRight: '1px solid rgba(74,144,217,0.2)' }}
      >
        {/* Status indicator */}
        <div
          className="rounded-lg px-3 py-2 text-xs"
          style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(74,144,217,0.3)' }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-2 h-2 rounded-full"
              style={{ background: isLoading ? '#e67e22' : '#27ae60' }}
            />
            <span className="text-gray-300 font-semibold">
              {isLoading ? 'Executing...' : 'Ready'}
            </span>
          </div>
          <p className="text-gray-500">AR4 6-DOF Robot Arm</p>
        </div>

        <JointStatePanel jointAngles={jointAngles} targetAngles={targetAngles} />

        {/* Quick pose buttons */}
        <div
          className="rounded-lg p-3 space-y-2"
          style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(74,144,217,0.3)' }}
        >
          <h3 className="text-xs font-semibold text-blue-300 uppercase tracking-wider">
            Quick Poses
          </h3>
          {['home', 'ready', 'wave', 'inspect'].map((pose) => (
            <button
              key={pose}
              onClick={() => sendInstruction(`Execute the ${pose} pose`)}
              disabled={isLoading}
              className="block w-full text-left text-xs px-2 py-1.5 rounded capitalize transition-colors"
              style={{
                background: 'rgba(74,144,217,0.1)',
                border: '1px solid rgba(74,144,217,0.2)',
                color: isLoading ? '#6b7280' : '#93c5fd',
                cursor: isLoading ? 'not-allowed' : 'pointer',
              }}
              onMouseEnter={(e) => {
                if (!isLoading)
                  (e.currentTarget as HTMLButtonElement).style.background =
                    'rgba(74,144,217,0.25)';
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.background =
                  'rgba(74,144,217,0.1)';
              }}
            >
              {pose}
            </button>
          ))}
        </div>
      </div>

      {/* Center: 3D canvas */}
      <div className="flex-1 relative">
        <SimulatorCanvas
          jointAngles={jointAngles}
          targetAngles={targetAngles}
          onAnglesUpdate={updateJointAngles}
        />

        {/* Overlay label */}
        <div
          className="absolute top-3 left-3 px-3 py-1.5 rounded text-xs font-mono"
          style={{
            background: 'rgba(0,0,0,0.6)',
            border: '1px solid rgba(74,144,217,0.3)',
            color: '#4a90d9',
            pointerEvents: 'none',
          }}
        >
          AR4 Simulator · Drag to orbit · Scroll to zoom
        </div>
      </div>

      {/* Right panel: instruction input + chat */}
      <div
        className="flex flex-col"
        style={{ width: '320px', minWidth: '320px', borderLeft: '1px solid rgba(74,144,217,0.2)' }}
      >
        <div className="flex-1 p-3 min-h-0">
          <InstructionPanel
            onSendInstruction={sendInstruction}
            messages={messages}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
