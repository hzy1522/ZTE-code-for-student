"""
Agent 派生类实现

选手需要实现自己的 Agent，继承 BaseAgent 并实现 act 方法。

改进版本：
1. 优化 system prompt，详细说明坐标系统
2. 利用历史消息帮助模型理解任务进度
3. 更好的输出格式引导
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional

from agent_base import (
    BaseAgent, AgentInput, AgentOutput,
    ACTION_CLICK, ACTION_SCROLL, ACTION_TYPE, ACTION_OPEN, ACTION_COMPLETE,
    VALID_ACTIONS, UsageInfo
)

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    GUI Agent 实现类

    继承 BaseAgent，实现根据用户指令和当前截图生成动作。

    改进点：
    1. 优化 prompt，引导模型更准确地理解界面和坐标
    2. 利用历史动作帮助模型理解当前进度
    3. 增强输出格式解析的鲁棒性
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化 Agent"""
        super().__init__(config)
        self._last_action = None
        self._last_params = None

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent 核心方法：根据输入生成动作

        Args:
            input_data: AgentInput 包含当前轮所有信息

        Returns:
            AgentOutput 包含动作和参数
        """
        # 1. 生成发送给大模型的 messages
        messages = self.generate_messages(input_data)

        # 2. 调用大模型 API
        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content

            # 3. 提取 Token 使用信息
            usage = self.extract_usage_info(response)

            # 4. 解析输出为 action 和 parameters
            action, parameters = self._parse_output(raw_output)

            # 5. 记录上一步动作，帮助调试
            self._last_action = action
            self._last_params = parameters

            logger.info(f"[Agent] Parsed action: {action}, params: {parameters}")
            logger.debug(f"[Agent] Raw output: {raw_output[:500]}")

            return AgentOutput(
                action=action,
                parameters=parameters,
                raw_output=raw_output,
                usage=usage
            )

        except Exception as e:
            logger.error(f"[Agent] Error calling API: {e}")
            return AgentOutput(
                action="",
                parameters={},
                raw_output=f"Error: {str(e)}"
            )

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """
        生成发给大模型的 messages

        改进版本：包含历史消息帮助模型理解上下文
        """
        # 构建系统提示
        system_prompt = self._build_system_prompt(input_data.instruction)

        # 构建当前轮消息
        current_content = []

        # 添加任务指令
        current_content.append({
            "type": "text",
            "text": f"用户任务: {input_data.instruction}"
        })

        # 添加历史动作摘要
        if input_data.history_actions:
            history_text = self._build_history_summary(input_data.history_actions)
            current_content.append({
                "type": "text",
                "text": f"历史动作摘要:\n{history_text}"
            })

        # 添加步骤计数
        current_content.append({
            "type": "text",
            "text": f"当前是第 {input_data.step_count} 步操作"
        })

        # 添加截图
        current_content.append({
            "type": "image_url",
            "image_url": {"url": self._encode_image(input_data.current_image)}
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_content}
        ]

        return messages

    def _build_system_prompt(self, instruction: str) -> str:
        """
        构建系统提示词

        改进版本：更详细地说明坐标系统和输出格式
        """
        return """你是一个手机GUI操作助手，负责根据用户指令和当前屏幕截图，完成手机操作任务。

## 任务说明
你将看到手机屏幕截图，需要根据用户指令执行正确的操作。

## 坐标系统 - 非常重要！
屏幕坐标范围是 [0, 1000]，即：
- 左上角坐标是 [0, 0]
- 右下角坐标是 [1000, 1000]
- x轴从左到右，y轴从上到下
- 当你需要点击时，必须输出目标位置的精确坐标

## 操作类型
1. CLICK - 点击操作：需要输出目标点的 [x, y] 坐标
2. TYPE - 输入文本：需要输出要输入的文本内容
3. SCROLL - 滚动操作：需要输出起始点和结束点的坐标
4. OPEN - 打开应用：需要输出应用名称
5. COMPLETE - 任务完成：不需要任何参数

## 输出格式 - 必须严格遵守！
每行只输出一个动作，格式如下：

点击搜索框：CLICK:[[400, 150]]
输入文字：TYPE:["北京"]
打开应用：OPEN:["微信"]
滚动屏幕：SCROLL:[[300, 800], [300, 200]]
任务完成：COMPLETE:[]

## 坐标获取技巧
- 仔细观察截图中的可点击元素（按钮、输入框、图标等）
- 估计元素中心点的坐标
- 坐标必须是 [x, y] 格式的归一化坐标（0-1000范围）
- 不要猜测，使用你观察到的实际位置

## 任务执行策略
1. 先理解用户想要完成什么任务
2. 观察当前屏幕显示了什么
3. 确定下一步应该执行什么操作
4. 输出对应的动作和参数

开始分析屏幕并执行任务！"""

    def _build_history_summary(self, history_actions: List[Dict[str, Any]]) -> str:
        """
        构建历史动作摘要

        帮助模型理解已经完成了哪些步骤，避免重复操作
        """
        if not history_actions:
            return "（无历史动作）"

        summary_parts = []
        for i, action_info in enumerate(history_actions[-5:], 1):  # 只取最近5步
            action = action_info.get('action', 'UNKNOWN')
            params = action_info.get('parameters', {})

            if action == ACTION_CLICK:
                point = params.get('point', [0, 0])
                summary_parts.append(f"{i}. CLICK 点击位置 {point}")
            elif action == ACTION_TYPE:
                text = params.get('text', '')[:20]
                summary_parts.append(f"{i}. TYPE 输入 '{text}...'")
            elif action == ACTION_OPEN:
                app = params.get('app_name', '')
                summary_parts.append(f"{i}. OPEN 打开 {app}")
            elif action == ACTION_SCROLL:
                start = params.get('start_point', [0, 0])
                end = params.get('end_point', [0, 0])
                summary_parts.append(f"{i}. SCROLL 从 {start} 到 {end}")
            elif action == ACTION_COMPLETE:
                summary_parts.append(f"{i}. COMPLETE 任务完成")
            else:
                summary_parts.append(f"{i}. {action} {params}")

        return "\n".join(summary_parts)

    def _parse_output(self, raw_output: str) -> tuple:
        """
        解析大模型原始输出，提取动作和参数

        支持多种输出格式的解析：
        1. 标准格式: "CLICK:[[x, y]]"
        2. 函数调用格式: "CLICK(point=\"<point>x y</point>\")"
        3. Action 章节格式: "Action: CLICK(...)"
        """
        raw_output = raw_output.strip()

        # 方式1: 函数调用格式 (带 <point> 标签的)
        func_patterns = {
            'CLICK': r'click\s*\(\s*point\s*=\s*"?\s*<point>\s*([\d.]+)\s+([\d.]+)\s*</point>\s*"?\s*\)',
            'TYPE': r'type\s*\(\s*content\s*=\s*["\']([^"\']+)["\']\s*\)',
            'SCROLL': r'scroll\s*\(\s*start_point\s*=\s*"?\s*<point>\s*([\d.]+)\s+([\d.]+)\s*</point>\s*"?\s*,\s*end_point\s*=\s*"?\s*<point>\s*([\d.]+)\s+([\d.]+)\s*</point>\s*"?\s*\)',
            'OPEN': r'open\s*\(\s*app_name\s*=\s*["\']([^"\']+)["\']\s*\)',
            'COMPLETE': r'complete\s*\(\s*\)',
        }

        for action_name, pattern in func_patterns.items():
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                params = self._parse_funcall_params(action_name, match)
                return action_name, params

        # 方式2: 赛题指定格式 "ACTION:[[...]]" 或 "ACTION:[...]"
        # 特殊处理 SCROLL，因为包含嵌套括号
        scroll_match = re.search(r'SCROLL:\s*(\[\[[^\]]*\],\s*\[[^\]]*\]\])', raw_output.upper())
        if scroll_match:
            param_str = scroll_match.group(1)
            params = self._parse_params(ACTION_SCROLL, param_str)
            if params is not None:
                return ACTION_SCROLL, params

        for action in VALID_ACTIONS:
            if action == ACTION_SCROLL:
                continue
            pattern = rf'{action}:\s*(\[[^\]]*\]|\{{[^{{]*\}}|\S+)'
            upper_match = re.search(pattern, raw_output.upper())
            if upper_match:
                start_pos = upper_match.start(1)
                end_pos = upper_match.end(1)
                param_str = raw_output[start_pos:end_pos]
                params = self._parse_params(action, param_str)
                if params is not None:
                    return action, params

        # 方式3: 从 "Action: CLICK(...)" 格式提取
        action_match = re.search(r'Action:\s*(\w+)', raw_output, re.IGNORECASE)
        if action_match:
            action_name = action_match.group(1).upper()
            if action_name in VALID_ACTIONS:
                params = self._extract_action_params(raw_output, action_name)
                if params:
                    return action_name, params
                return action_name, {}

        # 解析失败
        logger.warning(f"[Agent] Failed to parse output: {raw_output[:200]}")
        return "", {}

    def _parse_params(self, action: str, param_str: str) -> Optional[Dict[str, Any]]:
        """
        根据动作类型解析参数字符串
        """
        try:
            if action == ACTION_CLICK:
                param_str = param_str.strip('[]')
                coords = json.loads(f"[{param_str}]")
                if len(coords) == 2:
                    return {"point": coords}
                coords = [float(x.strip()) for x in param_str.replace('[', '').replace(']', '').split(',')]
                if len(coords) == 2:
                    return {"point": coords}

            elif action == ACTION_TYPE:
                text = param_str.strip("[]'\"")
                return {"text": text}

            elif action == ACTION_OPEN:
                text = param_str.strip("[]'\"")
                return {"app_name": text}

            elif action == ACTION_SCROLL:
                try:
                    data = json.loads(param_str)
                    if isinstance(data, list) and len(data) == 2:
                        start, end = data
                        if len(start) == 2 and len(end) == 2:
                            return {"start_point": start, "end_point": end}
                except:
                    param_str = param_str.strip('[]')
                    parts = param_str.split('],')
                    if len(parts) >= 2:
                        start = json.loads(parts[0] + ']')
                        end = json.loads('[' + parts[1])
                        if len(start) == 2 and len(end) == 2:
                            return {"start_point": start, "end_point": end}

            elif action == ACTION_COMPLETE:
                return {}

        except Exception as e:
            logger.debug(f"[Agent] Failed to parse params for {action}: {e}")
        return None

    def _extract_action_params(self, raw_output: str, action: str) -> Dict[str, Any]:
        """
        从原始输出中提取动作参数
        """
        params = {}

        if action == ACTION_CLICK:
            point_match = re.search(r'<point>\s*([\d.]+)\s+([\d.]+)\s*</point>', raw_output)
            if point_match:
                params["point"] = [int(float(point_match.group(1))), int(float(point_match.group(2)))]
            else:
                point_match = re.search(r'point\s*=\s*[\'"]?([^\)\'"\s]+)[\'"]?', raw_output)
                if point_match:
                    coords_str = point_match.group(1)
                    coords_match = re.findall(r'\d+', coords_str)
                    if len(coords_match) >= 2:
                        params["point"] = [int(coords_match[0]), int(coords_match[1])]

        elif action == ACTION_TYPE:
            text_match = re.search(r'content\s*=\s*["\']([^"\']+)["\']', raw_output)
            if text_match:
                params["text"] = text_match.group(1)

        elif action == ACTION_OPEN:
            app_match = re.search(r'app_name\s*=\s*["\']([^"\']+)["\']', raw_output)
            if app_match:
                params["app_name"] = app_match.group(1)

        elif action == ACTION_SCROLL:
            matches = re.findall(r'<point>\s*([\d.]+)\s+([\d.]+)\s*</point>', raw_output)
            if len(matches) >= 2:
                params["start_point"] = [int(float(matches[0][0])), int(float(matches[0][1]))]
                params["end_point"] = [int(float(matches[1][0])), int(float(matches[1][1]))]

        elif action == ACTION_COMPLETE:
            params = {}

        return params

    def _parse_coord_string(self, coord_str: str) -> List[int]:
        """解析坐标字符串"""
        coord_str = coord_str.strip('[]\s')
        if ',' in coord_str:
            parts = coord_str.split(',')
        else:
            parts = coord_str.split()
        try:
            return [int(float(p.strip())) for p in parts if p.strip()]
        except:
            return [0, 0]

    def _parse_funcall_params(self, action: str, match: re.Match) -> Dict[str, Any]:
        """从正则匹配结果中提取函数调用参数"""
        if action == ACTION_CLICK:
            return {"point": [int(float(match.group(1))), int(float(match.group(2)))]}
        elif action == ACTION_TYPE:
            return {"text": match.group(1)}
        elif action == ACTION_OPEN:
            return {"app_name": match.group(1)}
        elif action == ACTION_SCROLL:
            return {
                "start_point": [int(float(match.group(1))), int(float(match.group(2)))],
                "end_point": [int(float(match.group(3))), int(float(match.group(4)))]
            }
        elif action == ACTION_COMPLETE:
            return {}
        return {}

    def reset(self):
        """重置 Agent 状态"""
        self._last_action = None
        self._last_params = None
