import { useContext } from 'react';
import { AgentContext } from '../components/AgentProvider';
import { Ar4RobotAgentAgentOptionsProxy } from '../generated/agent/options-proxy.gen';

export const useAgent = (): Ar4RobotAgentAgentOptionsProxy => {
  const optionsProxy = useContext(AgentContext);

  if (!optionsProxy) {
    throw new Error('useAgent must be used within a AgentProvider');
  }

  return optionsProxy;
};
