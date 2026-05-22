import { ApplicationStage } from './stages/application-stage.js';
import { App } from ':ar4-sim/common-constructs';

const app = new App();

new ApplicationStage(app, 'ar4-sim-infra-sandbox', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});

app.synth();
