# AR4 Robot Simulator

A full-stack web application for controlling and visualising the [AR4 6-DOF robot arm](https://www.anninrobotics.com/) using natural language instructions. Type a command in plain English — a Strands AI agent translates it into ROS2 commands and animates the robot in a 3D browser simulator.

![AR4 Simulator](https://img.shields.io/badge/status-active-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (React)                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Joint State │  │  3D Viewer   │  │  Instruction  │  │
│  │    Panel     │  │  (Three.js)  │  │  Chat Panel   │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                           │                  │           │
│                    useSimulator hook          │           │
│                           │                  │           │
└───────────────────────────┼──────────────────┼───────────┘
                            │                  │
                     joint angles        SigV4 fetch
                     (lerp animation)         │
                                              ▼
                              ┌───────────────────────────┐
                              │  Strands Agent (Python)    │
                              │  AWS Bedrock AgentCore     │
                              │                            │
                              │  Tools:                    │
                              │  • move_joint              │
                              │  • execute_pose            │
                              │  • open/close_gripper      │
                              │  • rotate_base             │
                              │  • get_joint_states        │
                              └───────────────────────────┘
```

**Infrastructure** (AWS CDK):
- CloudFront + S3 — static website hosting
- Cognito User Pool + Identity Pool — authentication
- Bedrock AgentCore Runtime — containerised Strands agent (ARM64)
- WAF — web application firewall on the CloudFront distribution
- AppConfig — runtime configuration delivery

---

## Monorepo Structure

```
packages/
├── sim-for-ar4/              # React frontend (TypeScript)
│   └── src/
│       ├── components/
│       │   ├── Ar4Simulator/ # 3D viewer, joint panel, chat UI
│       │   ├── AppLayout/    # Cloudscape shell + navigation
│       │   ├── CognitoAuth/  # OIDC auth wrapper
│       │   └── RuntimeConfig/# Runtime config provider
│       ├── hooks/            # useAgent, useSigV4, useRuntimeConfig
│       └── routes/           # TanStack Router file-based routes
│
├── ar4-robot-agent/          # Strands agent backend (Python)
│   └── ar4_robot_agent/
│       └── ar4_sim_ar4_robot_agent/
│           └── agent/
│               ├── agent.py  # Tools + system prompt
│               └── main.py   # FastAPI streaming server
│
├── common/
│   ├── constructs/           # Shared CDK constructs
│   │   └── src/
│   │       ├── core/         # StaticWebsite, UserIdentity, RuntimeConfig
│   │       └── app/          # SimForAr4, Ar4RobotAgentAgent
│   └── agent_connection/     # Shared Python session/runtime-config helpers
│
└── infra/                    # CDK app entry point
    └── src/
        ├── main.ts
        ├── stacks/application-stack.ts
        └── stages/application-stage.ts
```

---

## Prerequisites

- Node.js 20+, pnpm
- Python 3.14+, [uv](https://docs.astral.sh/uv/getting-started/installation/)
- AWS CLI configured with credentials that have Bedrock access
- Docker (for building the agent container image)

---

## Running Locally

Start both processes in separate terminals:

**Terminal 1 — Agent backend**
```bash
# Set AWS credentials so the agent can call Bedrock
export AWS_PROFILE=your-profile   # or AWS_ACCESS_KEY_ID / SECRET / SESSION_TOKEN
export AWS_REGION=us-east-1

npx nx run ar4_sim.ar4_robot_agent:agent-serve-local
# Starts FastAPI on http://localhost:8081
```

**Terminal 2 — Frontend**
```bash
npx nx run @ar4-sim/sim-for-ar4:serve-local
# Opens http://localhost:4200
```

In `serve-local` mode Cognito auth is bypassed and the frontend points directly to `http://localhost:8081`.

---

## Deploying to AWS

```bash
# First time only — bootstrap CDK in your account/region
npx nx run @ar4-sim/infra:bootstrap

# Build everything and deploy
npx nx run @ar4-sim/infra:deploy
```

The CloudFront URL is printed as `DistributionDomainName` in the stack outputs.

To load the deployed runtime config for local development against the real backend:
```bash
npx nx run @ar4-sim/sim-for-ar4:load:runtime-config
```

---

## How It Works

1. You type a natural language instruction (e.g. *"wave at me"*) in the chat panel.
2. The frontend POSTs it to the Strands agent via a SigV4-signed streaming request.
3. The agent calls its ROS2 tools (`execute_pose`, `move_joint`, etc.) and emits a ` ```ros2 ``` ` fenced JSON block in its response.
4. The frontend parses the block, updates `targetAngles` state, and the `useFrame` loop smoothly lerps the Three.js joint groups toward the new angles.

### AR4 Joints

| Joint | Axis | Range |
|-------|------|-------|
| Base | Y (turntable) | ±180° |
| Shoulder | X | −90° to 90° |
| Elbow | X | −135° to 135° |
| Wrist Pitch | X | −90° to 90° |
| Wrist Roll | Z | ±180° |
| Gripper | ±X spread | 0% (closed) – 100% (open) |

### Named Poses

`home` · `ready` · `pick_up` · `place_down` · `wave` · `inspect` · `rest`

---

## Common Tasks

```bash
# Build all packages
pnpm build

# Lint and auto-fix
pnpm lint

# Run tests
pnpm nx run-many --target test --all

# Sync TypeScript project references
pnpm nx sync
```

---

## Next Steps

### Real Simulator Integration

The current 3D viewer is a lightweight browser renderer built with Three.js. For higher-fidelity simulation with physics, collision detection, and sensor data, the agent can be connected to a proper robotics simulator:

#### Option A — Gazebo via rosbridge

[Gazebo](https://gazebosim.org/) is the standard ROS2 simulator. The frontend can communicate with it over WebSocket using [`rosbridge_suite`](https://github.com/RobotWebTools/rosbridge_suite) and [`roslibjs`](https://github.com/RobotWebTools/roslibjs):

1. Run a ROS2 + Gazebo environment with the AR4 URDF loaded
2. Start `rosbridge_websocket` on port 9090
3. Replace the Three.js joint updates in `useSimulator.ts` with `ros.publish()` calls to `/ar4/joint_trajectory_controller/command`
4. Subscribe to `/ar4/joint_states` to read back actual positions for the joint panel

```ts
// Example roslibjs integration
import ROSLIB from 'roslib';
const ros = new ROSLIB.Ros({ url: 'ws://localhost:9090' });
const cmdTopic = new ROSLIB.Topic({ ros, name: '/ar4/joint_trajectory_controller/command', messageType: 'trajectory_msgs/JointTrajectory' });
cmdTopic.publish(new ROSLIB.Message({ joint_names: ['base', ...], points: [{ positions: [...], time_from_start: { secs: 1 } }] }));
```

#### Option B — NVIDIA Isaac Sim

[Isaac Sim](https://developer.nvidia.com/isaac/sim) provides GPU-accelerated physics and photorealistic rendering. It exposes a ROS2 bridge, so the same rosbridge approach above applies. Useful for training perception models alongside the control agent.

#### Option C — MuJoCo / PyBullet (lightweight)

For a Python-native physics simulation without a full ROS2 stack, [MuJoCo](https://mujoco.org/) or [PyBullet](https://pybullet.org/) can be embedded directly in the agent backend. The agent tools would drive the simulator instead of returning JSON, and stream back rendered frames or joint feedback.

---

## Acknowledgements

- [Annin Robotics](https://www.anninrobotics.com/) — AR4 robot arm design
- [稚晖君](https://github.com/peng-zhihui) — original Dummy Robot inspiration
- [AWS Nx Plugin](https://awslabs.github.io/nx-plugin-for-aws/) — project scaffolding
- [Strands Agents](https://github.com/strands-agents/sdk-python) — agent framework

## License

MIT
