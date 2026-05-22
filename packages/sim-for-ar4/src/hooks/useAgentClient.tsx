import { Ar4RobotAgentAgent } from '../generated/agent/client.gen';
import { AgentClientContext } from '../components/AgentProvider';
import { useContext } from 'react';

export const useAgentClient = (): Ar4RobotAgentAgent => {
  const client = useContext(AgentClientContext);

  if (!client) {
    throw new Error('useAgentClient must be used within a AgentProvider');
  }

  return client;
};
