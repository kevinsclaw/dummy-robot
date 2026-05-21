# Dummy Robot V2

开源 6-DOF 桌面机械臂 — 基于[稚晖君 Dummy](https://github.com/peng-zhihui/Dummy-Robot) V2 版本的 LLM 智能控制方案。

## 项目结构

```
dummy-robot/
├── dummy-sim/       # 仿真环境 (运动学 + 3D 可视化)
├── dummy-demo/      # 实机控制 (串口驱动 + LLM 规划 + RGBD 视觉)
└── README.md
```

## 硬件

| 组件 | 型号 |
|------|------|
| 机械臂 | Dummy V2 (6-DOF + 夹爪) |
| 主控 | STM32F405 (REF 控制板) |
| 深度相机 | Orbbec (RGB 1080p + 深度 640x400) |
| 通信 | USB CDC 串口 (ASCII 协议) |

## 快速开始

### 仿真

```bash
cd dummy-sim
pip install numpy matplotlib
python visualize.py
```

### 实机控制

```bash
cd dummy-demo
pip install pyserial numpy opencv-python

# 硬件连通性测试
python main.py --test

# 交互控制
python main.py
```

> ⚠️ USB Type-C 有正反面 — **翻面**才连接 STM32 CDC。

## 功能

- **运动学引擎** — 标准 DH 正/逆运动学，经过固件源码验证
- **串口控制** — 关节空间 & 笛卡尔空间运动，夹爪开合
- **RGBD 视觉** — Orbbec 深度相机，颜色检测 + 定位
- **LLM 规划** — AWS Bedrock Claude 自然语言→抓取任务 (WIP)

## 安全注意事项

- 开机顺序: 上电 → `!HOME` → `!START`
- **关机顺序: `!RESET` (折叠) → `!DISABLE` → 断电**
- 不要在展开状态断电，臂会自由坠落

## 致谢

- [稚晖君](https://github.com/peng-zhihui) — 原始设计
- 木子晓汶 — PCB 二次开发
- 任同学 — V2 版本整理

## License

MIT
# my-project

✨ Your new, shiny [Nx workspace](https://nx.dev) has been successfully created! ✨.

[Learn more about this workspace setup and the @aws/nx-plugin](https://awslabs.github.io/nx-plugin-for-aws). Now, let's get you up to speed!

## Install Nx Console

Nx Console is an editor extension that enriches your developer experience. It lets you run tasks, generate code, and improves code autocompletion in your IDE. It is available for VSCode and IntelliJ.

[Install Nx Console &raquo;](https://nx.dev/getting-started/editor-setup?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects)

## Available generators

The following list of generators are what is currently available in the `@aws/nx-plugin`:

- **connection**: Integrates a source project with a target project

- **license**: Add LICENSE files and configure source code licence headers

- **py#fast-api**: Generates a FastAPI Python project

- **py#lambda-function**: Adds a lambda function to a python project

- **py#mcp-server**: Generate a Python Model Context Protocol (MCP) server for providing context to Large Language Models

- **py#project**: Generates a Python project

- **py#strands-agent**: Add a Strands Agent to a Python project

- **terraform#project**: Generates a Terraform project

- **ts#astro-docs**: Generates an Astro + Starlight documentation site with localisation, snippets, blog, and optional automated documentation translation

- **ts#infra**: Generates a cdk application

- **ts#lambda-function**: Generate a TypeScript lambda function

- **ts#mcp-server**: Generate a TypeScript Model Context Protocol (MCP) server for providing context to Large Language Models

- **ts#nx-generator**: Generator for adding an Nx Generator to an existing TypeScript project

- **ts#nx-plugin**: Generate an Nx Plugin of your own! Build custom generators automatically made available for AI vibe-coding via MCP

- **ts#project**: Generates a TypeScript project

- **ts#react-website**: Generates a React static website

- **ts#react-website#auth**: Adds auth to an existing React website

- **ts#smithy-api**: Create an API using Smithy and the Smithy TypeScript Server SDK

- **ts#strands-agent**: Add a Strands Agent to a TypeScript project

- **ts#trpc-api**: creates a trpc backend

- **ts#rdb**: Create a relational database project

You also have the option of using additional [commmunity plugins](https://nx.dev/plugin-registry) as needed.

## Invoking a generator

```sh
pnpm nx g @aws/nx-plugin:<generator-name>
```

Alternatively you can use the Nx IDE plugin to invoke your generators.

Refer to the [full documentation](https://awslabs.github.io/nx-plugin-for-aws) for additional guidance for each generator.

## Common tasks

### Build a single project

```sh
pnpm nx build <project-name>
```

### Build all projects

```sh
pnpm nx run-many --target build --all
# or
pnpm build
```

### Run arbitrary task

```sh
pnpm nx <target> <project-name>
```

### Lint (and fix) all projects

```sh
pnpm nx run-many --target lint --configuration=fix --all
# or
pnpm lint
```

## Test all projects (and update snapshots)

```sh
pnpm nx run-many --target test --all --update
```

These targets are either [inferred automatically](https://nx.dev/concepts/inferred-tasks?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects) or defined in the `project.json` or `package.json` files.

[More about running tasks in the Nx docs &raquo;](https://nx.dev/features/run-tasks?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects)

## Keep TypeScript project references up to date

Nx automatically updates TypeScript [project references](https://www.typescriptlang.org/docs/handbook/project-references.html) in `tsconfig.json` files to ensure they remain accurate based on your project dependencies (`import` statements). This sync is automatically done when running tasks such as `build`, which require updated references to function correctly.

To manually trigger the process to sync the project graph dependencies information to the TypeScript project references, run the following command:

```sh
pnpm nx sync
```

You can enforce that the TypeScript project references are always in the correct state when running in CI by adding a step to your CI job configuration that runs the following command:

```sh
pnpm nx sync:check
```

[Learn more about nx sync](https://nx.dev/reference/nx-commands#sync)

## Set up CI!

Use the following command to configure a CI workflow for your workspace:

```sh
pnpm nx g ci-workflow
```

[Learn more about Nx on CI](https://nx.dev/ci/intro/ci-with-nx#ready-get-started-with-your-provider?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects)

## Useful links

Learn more:

- [@aws/nx-plugin quick-start](https://awslabs.github.io/nx-plugin-for-aws/en/get_started/quick-start/)
- [@aws/nx-plugin AI dungeon game](https://awslabs.github.io/nx-plugin-for-aws/en/get_started/tutorials/dungeon-game/overview/)
- [What are Nx plugins?](https://nx.dev/concepts/nx-plugins?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects)
- [Learn about Nx on CI](https://nx.dev/ci/intro/ci-with-nx?utm_source=nx_project&utm_medium=readme&utm_campaign=nx_projects)
