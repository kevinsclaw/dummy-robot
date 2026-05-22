import {
  Ar4RobotAgentAgent,
  SimForAr4,
  UserIdentity,
} from ':ar4-sim/common-constructs';
import { Stack, StackProps } from 'aws-cdk-lib';
import { Construct } from 'constructs';

export class ApplicationStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const userIdentity = new UserIdentity(this, 'UserIdentity');
    const ar4Agent = new Ar4RobotAgentAgent(this, 'Ar4RobotAgent');

    // Grant authenticated users permission to invoke the agent
    ar4Agent.grantInvokeAccess(userIdentity.identityPool.authenticatedRole);

    new SimForAr4(this, 'SimForAR4');
  }
}
