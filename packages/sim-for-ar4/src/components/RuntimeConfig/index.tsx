import { Spinner } from '../spinner';
import React, {
  createContext,
  PropsWithChildren,
  useEffect,
  useState,
} from 'react';

// Consider specifying types if desired
export type IRuntimeConfig = any;

/**
 * Context for storing the runtimeConfig.
 */
export const RuntimeConfigContext = createContext<IRuntimeConfig | undefined>(
  undefined,
);

/**
 * Apply any overrides to point to local servers/resources here
 * for the serve-local target
 */
const applyOverrides = (runtimeConfig: IRuntimeConfig) => {
  if (import.meta.env.MODE === 'serve-local') {
    runtimeConfig.agentRuntimes = runtimeConfig.agentRuntimes ?? {};
    // The generated client appends /invocations itself, so the base URL must not include it
    runtimeConfig.agentRuntimes.Agent = 'http://localhost:8081';
    runtimeConfig.agentRuntimes.Ar4RobotAgentAgent = 'http://localhost:8081';
  }
  return runtimeConfig;
};

/**
 * Sets up the runtimeConfig.
 *
 * This assumes a runtime-config.json file is present at '/'.
 */
const RuntimeConfigProvider: React.FC<PropsWithChildren> = ({ children }) => {
  const [runtimeConfig, setRuntimeConfig] = useState<
    IRuntimeConfig | undefined
  >();
  useEffect(() => {
    (async () => {
      try {
        setRuntimeConfig(
          applyOverrides(await (await fetch('/runtime-config.json')).json()),
        );
      } catch {
        setRuntimeConfig(
          applyOverrides({ apis: {}, agentRuntimes: {} } as any),
        );
      }
    })();
  }, [setRuntimeConfig]);

  return runtimeConfig ? (
    <RuntimeConfigContext.Provider value={runtimeConfig}>
      {children}
    </RuntimeConfigContext.Provider>
  ) : (
    <Spinner />
  );
};

export default RuntimeConfigProvider;
