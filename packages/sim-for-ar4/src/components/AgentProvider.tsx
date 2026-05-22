import { createContext, FC, PropsWithChildren, useMemo } from 'react';
import { Ar4RobotAgentAgent } from '../generated/agent/client.gen';
import { Ar4RobotAgentAgentOptionsProxy } from '../generated/agent/options-proxy.gen';
import { useRuntimeConfig } from '../hooks/useRuntimeConfig';
import { useSigV4 } from '../hooks/useSigV4';

/**
 * Build an HTTP URL from a Bedrock AgentCore Runtime ARN
 */
function buildAgentCoreHttpUrl(agentRuntimeArn: string): string {
  const region = agentRuntimeArn.split(':')[3];
  return `https://bedrock-agentcore.${region}.amazonaws.com/runtimes/${encodeURIComponent(agentRuntimeArn)}`;
}

export const AgentContext = createContext<Ar4RobotAgentAgentOptionsProxy | undefined>(
  undefined,
);

export const AgentClientContext = createContext<Ar4RobotAgentAgent | undefined>(undefined);

const useCreateAgentClient = (): Ar4RobotAgentAgent => {
  const runtimeConfig = useRuntimeConfig();
  // Support both the generated key (agentcore.agentRuntimes.Ar4RobotAgentAgent)
  // and the legacy flat key (agentRuntimes.Agent)
  const agentRuntimeValue =
    runtimeConfig?.agentcore?.agentRuntimes?.Ar4RobotAgentAgent ??
    runtimeConfig?.agentRuntimes?.Ar4RobotAgentAgent ??
    runtimeConfig?.agentRuntimes?.Agent;
  const apiUrl = agentRuntimeValue?.startsWith('arn:')
    ? buildAgentCoreHttpUrl(agentRuntimeValue)
    : (agentRuntimeValue ?? 'http://localhost:8081');
  const sigv4Client = useSigV4();
  return useMemo(
    () =>
      new Ar4RobotAgentAgent({
        url: apiUrl,
        fetch: sigv4Client.fetch,
      }),
    [apiUrl, sigv4Client],
  );
};

export const AgentProvider: FC<PropsWithChildren> = ({ children }) => {
  const client = useCreateAgentClient();
  const optionsProxy = useMemo(
    () => new Ar4RobotAgentAgentOptionsProxy({ client }),
    [client],
  );

  return (
    <AgentClientContext.Provider value={client}>
      <AgentContext.Provider value={optionsProxy}>
        {children}
      </AgentContext.Provider>
    </AgentClientContext.Provider>
  );
};

export default AgentProvider;
