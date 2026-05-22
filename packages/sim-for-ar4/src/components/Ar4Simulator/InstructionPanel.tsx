import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Ros2Command } from './types';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  commands?: Ros2Command[];
  timestamp: Date;
}

interface InstructionPanelProps {
  onSendInstruction: (instruction: string) => void;
  messages: ChatMessage[];
  isLoading: boolean;
}

const EXAMPLE_INSTRUCTIONS = [
  'Move to home position',
  'Wave at me',
  'Pick up an object in front',
  'Rotate the base 90 degrees to the left',
  'Open the gripper fully',
  'Move to ready position slowly',
];

export default function InstructionPanel({
  onSendInstruction,
  messages,
  isLoading,
}: InstructionPanelProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSendInstruction(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className="flex flex-col h-full rounded-lg overflow-hidden"
      style={{ background: 'rgba(0,0,0,0.6)', border: '1px solid rgba(74,144,217,0.3)' }}
    >
      {/* Header */}
      <div
        className="px-4 py-3 flex items-center gap-2"
        style={{ borderBottom: '1px solid rgba(74,144,217,0.2)' }}
      >
        <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        <h3 className="text-sm font-semibold text-blue-300 uppercase tracking-wider">
          Strands Agent
        </h3>
        <span className="text-xs text-gray-500 ml-auto">AR4 Controller</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-xs text-gray-400 text-center py-2">
              Send a natural language instruction to control the AR4 robot arm.
            </p>
            <div className="space-y-2">
              <p className="text-xs text-gray-500 uppercase tracking-wider">Try:</p>
              {EXAMPLE_INSTRUCTIONS.map((ex) => (
                <button
                  key={ex}
                  onClick={() => {
                    setInput(ex);
                    inputRef.current?.focus();
                  }}
                  className="block w-full text-left text-xs px-3 py-2 rounded transition-colors"
                  style={{
                    background: 'rgba(74,144,217,0.1)',
                    border: '1px solid rgba(74,144,217,0.2)',
                    color: '#93c5fd',
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background =
                      'rgba(74,144,217,0.2)';
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.background =
                      'rgba(74,144,217,0.1)';
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className="max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed"
              style={
                msg.role === 'user'
                  ? {
                      background: 'rgba(74,144,217,0.3)',
                      border: '1px solid rgba(74,144,217,0.4)',
                      color: '#e2e8f0',
                    }
                  : {
                      background: 'rgba(39,174,96,0.15)',
                      border: '1px solid rgba(39,174,96,0.3)',
                      color: '#d1fae5',
                    }
              }
            >
              {msg.role === 'assistant' && (
                <div className="flex items-center gap-1 mb-1 opacity-60">
                  <span className="text-green-400">🤖</span>
                  <span className="text-green-400 font-semibold text-xs">Agent</span>
                </div>
              )}
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.commands && msg.commands.length > 0 && (
                <div className="mt-2 space-y-1">
                  {msg.commands.map((cmd, i) => (
                    <div
                      key={i}
                      className="text-xs font-mono px-2 py-1 rounded"
                      style={{
                        background: 'rgba(0,0,0,0.3)',
                        color: '#fbbf24',
                        border: '1px solid rgba(251,191,36,0.2)',
                      }}
                    >
                      <span className="text-gray-500">ROS2 </span>
                      {cmd.ros2_command}
                      {cmd.joint && <span className="text-blue-300"> {cmd.joint}</span>}
                      {cmd.angle_degrees !== undefined && (
                        <span className="text-green-300"> {cmd.angle_degrees}°</span>
                      )}
                      {cmd.pose_name && <span className="text-purple-300"> {cmd.pose_name}</span>}
                      {cmd.action && <span className="text-orange-300"> {cmd.action}</span>}
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-1 text-right opacity-40 text-xs">
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div
              className="rounded-lg px-3 py-2"
              style={{
                background: 'rgba(39,174,96,0.15)',
                border: '1px solid rgba(39,174,96,0.3)',
              }}
            >
              <div className="flex items-center gap-1">
                <span className="text-green-400">🤖</span>
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-green-400 animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div
        className="p-3"
        style={{ borderTop: '1px solid rgba(74,144,217,0.2)' }}
      >
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. Move to home position, then wave..."
            disabled={isLoading}
            rows={2}
            className="flex-1 resize-none text-xs rounded px-3 py-2 outline-none transition-colors"
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(74,144,217,0.3)',
              color: '#e2e8f0',
              fontFamily: 'inherit',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'rgba(74,144,217,0.7)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'rgba(74,144,217,0.3)';
            }}
            aria-label="Robot instruction input"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-3 py-2 rounded text-xs font-semibold transition-all self-end"
            style={{
              background:
                !input.trim() || isLoading
                  ? 'rgba(74,144,217,0.2)'
                  : 'rgba(74,144,217,0.8)',
              color: !input.trim() || isLoading ? '#6b7280' : '#fff',
              border: '1px solid rgba(74,144,217,0.4)',
              cursor: !input.trim() || isLoading ? 'not-allowed' : 'pointer',
            }}
            aria-label="Send instruction"
          >
            Send
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-1">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
