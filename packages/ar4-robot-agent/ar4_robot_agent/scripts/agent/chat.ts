/**
 * Minimal chat CLI for Ar4RobotAgentAgent (Python FastAPI / JSONL streaming).
 *
 * Uses the type-safe TypeScript client generated from the agent's OpenAPI
 * spec. Connects to the agent at `process.env.URL` (set by the Nx
 * `agent-chat` target).
 */
import { randomUUID } from 'node:crypto';
import { chatLoop, type ChatAdapter } from 'agent-chat-cli';
import { Ar4RobotAgentAgent } from './generated/client.gen.js';

const SESSION_ID_HEADER = 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id';

class Ar4RobotAgentAgentAdapter implements ChatAdapter {
  private client!: Ar4RobotAgentAgent;
  // AgentCore session IDs must be at least 33 characters.
  private readonly sessionId = randomUUID().replaceAll('-', '').padEnd(33, '0');

  async connect(url: string) {
    this.client = new Ar4RobotAgentAgent({
      url,
      fetch: (input, init) => {
        const headers = new Headers(init?.headers);
        headers.set(SESSION_ID_HEADER, this.sessionId);
        return fetch(input, { ...init, headers });
      },
    });
    return { agentName: 'Ar4RobotAgentAgent' };
  }

  async *sendMessage(text: string): AsyncIterable<string> {
    for await (const chunk of this.client.invoke({ message: text })) {
      if (typeof chunk.content === 'string') yield chunk.content;
    }
  }
}

await chatLoop(new Ar4RobotAgentAgentAdapter(), process.env.URL!);
