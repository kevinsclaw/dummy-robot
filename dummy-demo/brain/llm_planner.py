"""
LLM 意图解析模块

将自然语言指令解析为机械臂动作序列。

使用:
    planner = LLMPlanner(provider="openai")  # 或 "bedrock"

    # 解析指令
    action = planner.parse("把红色积木放到盘子里")
    print(action)
    # → RobotAction(type="pick_and_place", object="red_block", target="plate")

    # 带视觉上下文
    scene = [
        {"name": "red_block", "color": "red", "x": 150, "y": 80},
        {"name": "blue_block", "color": "blue", "x": 200, "y": 120},
        {"name": "plate", "color": "white", "x": 300, "y": 200},
    ]
    action = planner.parse("把红色的放到蓝色旁边", scene_objects=scene)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RobotAction:
    """解析出的机器人动作"""
    type: str                    # 动作类型
    object: Optional[str] = None       # 目标物体
    target: Optional[str] = None       # 放置目标
    params: Dict[str, Any] = field(default_factory=dict)  # 附加参数
    raw_text: str = ""                 # 原始文本
    confidence: float = 1.0            # 置信度

    def __str__(self):
        parts = [f"Action({self.type})"]
        if self.object:
            parts.append(f"obj={self.object}")
        if self.target:
            parts.append(f"target={self.target}")
        if self.params:
            parts.append(f"params={self.params}")
        return " ".join(parts)


# 动作类型定义
ACTION_TYPES = {
    "pick_and_place": "抓取物体放到目标位置",
    "pick_up": "抓起物体 (不放下)",
    "put_down": "放下当前持有的物体",
    "point_at": "指向某个物体",
    "push": "推动物体",
    "sort_by_color": "按颜色分拣",
    "stack": "堆叠物体",
    "count": "数物体数量",
    "wave": "挥手打招呼",
    "nod": "点头",
    "go_home": "回到初始位置",
    "stop": "紧急停止",
    "describe_scene": "描述当前场景",
}

SYSTEM_PROMPT = """你是一个机械臂控制器的指令解析器。

你需要将用户的自然语言指令解析为结构化的机器人动作。

可用的动作类型:
{action_types}

当前场景中的物体:
{scene_description}

规则:
1. 如果用户的指令不明确，推测最合理的意图
2. 物体用颜色+类型标识，如 "red_block", "blue_block", "plate"
3. 位置描述转为相对方位: "左边"→"left", "右边"→"right", "旁边"→"near"
4. 如果无法解析，type 设为 "unknown"
5. 一次只返回一个主要动作

你必须只返回一个 JSON 对象，格式:
{{
    "type": "动作类型",
    "object": "目标物体 (可选)",
    "target": "目标位置/物体 (可选)",
    "params": {{额外参数}},
    "confidence": 0.0-1.0
}}
"""


class LLMPlanner:
    """
    LLM 意图解析器

    Args:
        provider: "openai" | "bedrock" | "local"
        model: 模型名称
        api_key: API key (OpenAI)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key

        # 默认模型
        if model is None:
            if provider == "openai":
                self.model = "gpt-4o-mini"
            elif provider == "bedrock":
                self.model = "anthropic.claude-3-haiku-20240307-v1:0"
            else:
                self.model = "gpt-4o-mini"
        else:
            self.model = model

        self._client = None

    def parse(
        self,
        text: str,
        scene_objects: Optional[List[Dict]] = None,
    ) -> RobotAction:
        """
        解析自然语言指令

        Args:
            text: 用户指令文本
            scene_objects: 场景中的物体列表
                [{"name": "red_block", "color": "red", "x": 150, "y": 80}, ...]

        Returns:
            解析出的动作
        """
        # 先尝试规则匹配 (快速、无需 API)
        rule_action = self._rule_match(text)
        if rule_action:
            logger.info(f"Rule matched: {text} → {rule_action}")
            return rule_action

        # LLM 解析
        try:
            llm_action = self._llm_parse(text, scene_objects)
            logger.info(f"LLM parsed: {text} → {llm_action}")
            return llm_action
        except Exception as e:
            logger.error(f"LLM parse failed: {e}")
            return RobotAction(type="unknown", raw_text=text, confidence=0.0)

    def parse_batch(
        self,
        text: str,
        scene_objects: Optional[List[Dict]] = None,
    ) -> List[RobotAction]:
        """
        解析可能包含多步的指令
        如: "先把红色积木拿起来，然后放到蓝色旁边"
        """
        # 简单实现: 按关键词分割
        separators = ["然后", "接着", "再", "之后"]
        parts = [text]
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts

        actions = []
        for part in parts:
            part = part.strip()
            if part:
                action = self.parse(part, scene_objects)
                if action.type != "unknown":
                    actions.append(action)

        return actions if actions else [self.parse(text, scene_objects)]

    # ─── 规则匹配 ────────────────────────────────────────────

    def _rule_match(self, text: str) -> Optional[RobotAction]:
        """简单规则匹配 (快速路径，不需要 API)"""
        text_lower = text.lower().strip()

        # 停止
        if any(w in text_lower for w in ["停", "stop", "急停", "别动"]):
            return RobotAction(type="stop", raw_text=text)

        # 打招呼
        if any(w in text_lower for w in ["打招呼", "挥手", "hello", "嗨", "你好"]):
            return RobotAction(type="wave", raw_text=text)

        # 回零
        if any(w in text_lower for w in ["回去", "回零", "回家", "回原位", "home"]):
            return RobotAction(type="go_home", raw_text=text)

        # 点头
        if any(w in text_lower for w in ["点头", "点个头", "nod"]):
            return RobotAction(type="nod", raw_text=text)

        # 数数
        if any(w in text_lower for w in ["数一数", "几个", "多少个", "count"]):
            return RobotAction(type="count", raw_text=text)

        # 描述
        if any(w in text_lower for w in ["看看", "描述", "有什么", "describe"]):
            return RobotAction(type="describe_scene", raw_text=text)

        # 分拣
        if any(w in text_lower for w in ["分拣", "分类", "排列", "sort"]):
            colors = []
            for c in ["红", "蓝", "绿", "黄", "橙", "紫"]:
                if c in text:
                    color_map = {"红": "red", "蓝": "blue", "绿": "green",
                                 "黄": "yellow", "橙": "orange", "紫": "purple"}
                    colors.append(color_map[c])
            return RobotAction(
                type="sort_by_color",
                params={"colors": colors} if colors else {},
                raw_text=text,
            )

        # 堆叠
        if any(w in text_lower for w in ["摞", "叠", "堆", "stack"]):
            return RobotAction(type="stack", raw_text=text)

        # 抓取放置 (简单模式)
        obj = self._extract_color_object(text)
        target = self._extract_target(text)

        if obj and target:
            return RobotAction(
                type="pick_and_place", object=obj, target=target, raw_text=text
            )
        elif obj and any(w in text for w in ["拿起", "抓", "pick", "拿"]):
            return RobotAction(
                type="pick_up", object=obj, raw_text=text
            )
        elif any(w in text for w in ["放下", "松手", "put down"]):
            return RobotAction(
                type="put_down", target=target, raw_text=text
            )
        elif obj and any(w in text for w in ["指", "point"]):
            return RobotAction(
                type="point_at", object=obj, raw_text=text
            )
        elif obj and any(w in text for w in ["推", "push"]):
            return RobotAction(
                type="push", object=obj, target=target, raw_text=text
            )

        return None  # 规则匹配失败，交给 LLM

    def _extract_color_object(self, text: str) -> Optional[str]:
        """从文本中提取颜色+物体"""
        color_map = {
            "红色": "red", "红": "red",
            "蓝色": "blue", "蓝": "blue",
            "绿色": "green", "绿": "green",
            "黄色": "yellow", "黄": "yellow",
            "橙色": "orange", "橙": "orange",
            "紫色": "purple", "紫": "purple",
        }
        obj_map = {
            "积木": "block", "方块": "block", "块": "block",
            "球": "ball", "盒子": "box", "杯子": "cup",
        }

        color = None
        obj_type = "block"  # 默认积木

        for cn, en in color_map.items():
            if cn in text:
                color = en
                break

        for cn, en in obj_map.items():
            if cn in text:
                obj_type = en
                break

        if color:
            return f"{color}_{obj_type}"
        return None

    def _extract_target(self, text: str) -> Optional[str]:
        """从文本中提取目标位置"""
        target_map = {
            "盘子": "plate", "碟子": "plate",
            "盒子": "box", "箱子": "box",
            "左边": "left_zone", "左": "left_zone",
            "右边": "right_zone", "右": "right_zone",
            "中间": "center_zone",
        }

        # 检查颜色目标 ("放到蓝色旁边")
        color_map = {
            "红色": "red", "蓝色": "blue", "绿色": "green",
            "黄色": "yellow", "橙色": "orange", "紫色": "purple",
        }

        for cn, en in target_map.items():
            if cn in text:
                return en

        # "放到X旁边"
        for cn, en in color_map.items():
            if cn in text and "旁" in text:
                return f"near_{en}_block"

        return None

    # ─── LLM 解析 ────────────────────────────────────────────

    def _llm_parse(
        self, text: str, scene_objects: Optional[List[Dict]]
    ) -> RobotAction:
        """调用 LLM 解析"""

        # 构建场景描述
        if scene_objects:
            scene_desc = "\n".join(
                f"  - {obj['name']}: 颜色={obj.get('color','?')}, "
                f"位置=({obj.get('x',0)}, {obj.get('y',0)}) mm"
                for obj in scene_objects
            )
        else:
            scene_desc = "  (未知，需要相机检测)"

        # 构建 action types 描述
        action_desc = "\n".join(
            f"  - {k}: {v}" for k, v in ACTION_TYPES.items()
        )

        system = SYSTEM_PROMPT.format(
            action_types=action_desc,
            scene_description=scene_desc,
        )

        # 调用 LLM
        response = self._call_llm(system, text)

        # 解析 JSON
        try:
            # 处理可能的 markdown 包裹
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
                response = response.rsplit("```", 1)[0]

            data = json.loads(response)
            return RobotAction(
                type=data.get("type", "unknown"),
                object=data.get("object"),
                target=data.get("target"),
                params=data.get("params", {}),
                confidence=data.get("confidence", 0.8),
                raw_text=text,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {response} ({e})")
            return RobotAction(type="unknown", raw_text=text, confidence=0.0)

    def _call_llm(self, system: str, user: str) -> str:
        """调用 LLM API"""
        if self.provider == "openai":
            return self._call_openai(system, user)
        elif self.provider == "bedrock":
            return self._call_bedrock(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, system: str, user: str) -> str:
        """OpenAI API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_bedrock(self, system: str, user: str) -> str:
        """AWS Bedrock API"""
        try:
            import boto3
        except ImportError:
            raise ImportError("pip install boto3")

        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name="us-east-1")

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.1,
        })

        response = self._client.invoke_model(
            modelId=self.model,
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
