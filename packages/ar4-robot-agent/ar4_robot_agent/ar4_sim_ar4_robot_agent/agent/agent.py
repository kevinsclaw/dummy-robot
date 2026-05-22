from contextlib import contextmanager
from typing import Annotated

from strands import Agent, tool


@tool
def move_joint(
    joint: Annotated[
        str,
        "Joint name: base, shoulder, elbow, wrist_pitch, wrist_roll, or gripper",
    ],
    angle_degrees: Annotated[
        float,
        "Target angle in degrees. Valid ranges: base ±180, shoulder -90 to 90, "
        "elbow -135 to 135, wrist_pitch -90 to 90, wrist_roll ±180, gripper 0 to 100 (open %)",
    ],
    speed: Annotated[float, "Movement speed 0.0–1.0 (default 0.5)"] = 0.5,
) -> dict:
    """Move a single AR4 robot joint to the specified angle.

    Returns a ROS2 command dict that the simulator will execute.
    """
    valid_joints = {"base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"}
    if joint not in valid_joints:
        return {"error": f"Unknown joint '{joint}'. Valid joints: {sorted(valid_joints)}"}
    speed = max(0.0, min(1.0, speed))
    return {
        "ros2_command": "joint_trajectory",
        "joint": joint,
        "angle_degrees": angle_degrees,
        "speed": speed,
        "topic": "/ar4/joint_trajectory_controller/command",
    }


@tool
def execute_pose(
    pose_name: Annotated[
        str,
        "Named pose: home, ready, pick_up, place_down, wave, inspect, or rest",
    ],
    speed: Annotated[float, "Movement speed 0.0–1.0 (default 0.5)"] = 0.5,
) -> dict:
    """Move the AR4 robot to a predefined named pose.

    Returns a ROS2 command dict that the simulator will execute.
    """
    poses = {
        "home": {"base": 0, "shoulder": 0, "elbow": 0, "wrist_pitch": 0, "wrist_roll": 0, "gripper": 0},
        "ready": {"base": 0, "shoulder": -30, "elbow": 60, "wrist_pitch": -30, "wrist_roll": 0, "gripper": 0},
        "pick_up": {"base": 0, "shoulder": -60, "elbow": 90, "wrist_pitch": -30, "wrist_roll": 0, "gripper": 100},
        "place_down": {"base": 90, "shoulder": -45, "elbow": 75, "wrist_pitch": -30, "wrist_roll": 0, "gripper": 0},
        "wave": {"base": 45, "shoulder": -45, "elbow": 90, "wrist_pitch": 45, "wrist_roll": 0, "gripper": 0},
        "inspect": {"base": 0, "shoulder": -15, "elbow": 30, "wrist_pitch": -15, "wrist_roll": 0, "gripper": 50},
        "rest": {"base": 0, "shoulder": 0, "elbow": 0, "wrist_pitch": 0, "wrist_roll": 0, "gripper": 0},
    }
    if pose_name not in poses:
        return {"error": f"Unknown pose '{pose_name}'. Valid poses: {sorted(poses.keys())}"}
    return {
        "ros2_command": "named_pose",
        "pose_name": pose_name,
        "joint_angles": poses[pose_name],
        "speed": max(0.0, min(1.0, speed)),
        "topic": "/ar4/named_pose_controller/command",
    }


@tool
def open_gripper(
    percentage: Annotated[float, "How open the gripper should be, 0 (closed) to 100 (fully open)"] = 100.0,
) -> dict:
    """Open or partially open the AR4 gripper.

    Returns a ROS2 command dict that the simulator will execute.
    """
    return {
        "ros2_command": "gripper",
        "action": "open",
        "percentage": max(0.0, min(100.0, percentage)),
        "topic": "/ar4/gripper_controller/command",
    }


@tool
def close_gripper(
    force: Annotated[float, "Gripping force 0.0–1.0 (default 0.5)"] = 0.5,
) -> dict:
    """Close the AR4 gripper to grasp an object.

    Returns a ROS2 command dict that the simulator will execute.
    """
    return {
        "ros2_command": "gripper",
        "action": "close",
        "force": max(0.0, min(1.0, force)),
        "topic": "/ar4/gripper_controller/command",
    }


@tool
def rotate_base(
    angle_degrees: Annotated[float, "Rotation angle in degrees, ±180"],
    speed: Annotated[float, "Rotation speed 0.0–1.0 (default 0.5)"] = 0.5,
) -> dict:
    """Rotate the AR4 robot base to the specified angle.

    Returns a ROS2 command dict that the simulator will execute.
    """
    return {
        "ros2_command": "joint_trajectory",
        "joint": "base",
        "angle_degrees": max(-180.0, min(180.0, angle_degrees)),
        "speed": max(0.0, min(1.0, speed)),
        "topic": "/ar4/joint_trajectory_controller/command",
    }


@tool
def get_joint_states() -> dict:
    """Query the current joint states of the AR4 robot.

    Returns a ROS2 command dict requesting the current joint positions.
    """
    return {
        "ros2_command": "get_joint_states",
        "topic": "/ar4/joint_states",
    }


@contextmanager
def get_agent():
    yield Agent(
        name="Ar4RobotAgentAgent",
        description="AR4 robot arm controller that translates natural language instructions into ROS2 commands",
        system_prompt="""
You are an expert robotics controller for the AR4 6-DOF robot arm.
Your job is to translate natural language instructions into precise ROS2 commands
that control the AR4 robot in a simulator.

## AR4 Robot Joints
- **base**: Rotates the entire arm left/right (±180°)
- **shoulder**: Raises/lowers the upper arm (-90° to 90°)
- **elbow**: Bends the forearm (-135° to 135°)
- **wrist_pitch**: Tilts the wrist up/down (-90° to 90°)
- **wrist_roll**: Rotates the wrist (±180°)
- **gripper**: Opens/closes the end effector (0=closed, 100=fully open)

## Available Tools
- `move_joint`: Move a single joint to a specific angle
- `execute_pose`: Move to a named pose (home, ready, pick_up, place_down, wave, inspect, rest)
- `open_gripper`: Open the gripper to a percentage
- `close_gripper`: Close the gripper with a specified force
- `rotate_base`: Rotate the base joint
- `get_joint_states`: Query current joint positions

## CRITICAL OUTPUT FORMAT
After calling your tools, you MUST end your response with a JSON block containing
ALL the commands you executed. Use this exact format — no exceptions:

```ros2
[
  <paste each tool result dict here, comma-separated>
]
```

Example for "move to home":
```ros2
[
  {"ros2_command": "named_pose", "pose_name": "home", "joint_angles": {"base": 0, "shoulder": 0, "elbow": 0, "wrist_pitch": 0, "wrist_roll": 0, "gripper": 0}, "speed": 0.5, "topic": "/ar4/named_pose_controller/command"}
]
```

Example for moving a single joint:
```ros2
[
  {"ros2_command": "joint_trajectory", "joint": "base", "angle_degrees": 90, "speed": 0.5, "topic": "/ar4/joint_trajectory_controller/command"}
]
```

## Instructions
1. Parse the user's natural language instruction carefully
2. Call the appropriate tools to execute the movement
3. Use named poses when they match the intent (e.g., "go home" → execute_pose("home"))
4. For pick-and-place: open gripper → move to pick → close gripper → move to place → open gripper
5. Always end with the ```ros2 block containing all executed commands
6. Keep your explanation brief — one sentence before the block is enough
""",
        tools=[move_joint, execute_pose, open_gripper, close_gripper, rotate_base, get_joint_states],
    )
