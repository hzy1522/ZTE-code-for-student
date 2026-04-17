"""
Agent 派生类实现

选手需要实现自己的 Agent，继承 BaseAgent 并实现 act 方法。

通用策略版本：
1. 简洁的 prompt - 通用规则，不偏向特定用例
2. temperature=0 - 最小化随机性
3. 稳定的输出格式
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional

from agent_base import (
    BaseAgent, AgentInput, AgentOutput
)

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    GUI Agent 实现类

    通用策略：
    1. 简洁 prompt - 只包含必要的通用规则
    2. temperature=0 - 确定性输出
    3. 清晰的动作格式指导
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化 Agent"""
        super().__init__(config)
        self._last_action = None
        self._last_params = None

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """调用 API"""
        from openai import OpenAI

        client = OpenAI(
            base_url=self._api_url,
            api_key=self._api_key
        )

        logger.info(f"[API调用] model={self._model_id}")

        # temperature=0 确保确定性输出
        completion = client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=0,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )

        return completion

    def act(self, input_data: AgentInput) -> AgentOutput:
        """Agent 核心方法"""
        messages = self.generate_messages(input_data)

        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content
            usage = self.extract_usage_info(response)
            action, parameters = self._parse_output(raw_output)

            self._last_action = action
            self._last_params = parameters

            logger.info(f"[Agent] {action} {parameters}")

            return AgentOutput(
                action=action,
                parameters=parameters,
                raw_output=raw_output,
                usage=usage
            )

        except Exception as e:
            logger.error(f"[Agent] Error: {e}")
            return AgentOutput(action="", parameters={}, raw_output=f"Error: {str(e)}")

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """生成 messages"""
        system_prompt = self._build_system_prompt(input_data.instruction)

        current_content = [
            {"type": "text", "text": f"任务: {input_data.instruction}"},
            {"type": "text", "text": f"第{input_data.step_count}步"},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(input_data.current_image)}
            }
        ]

        # 添加简洁的历史信息
        if input_data.history_actions:
            history_text = "已完成: " + ", ".join([
                f"{h.get('action', '')}" for h in input_data.history_actions[-3:]
            ])
            current_content.append({"type": "text", "text": history_text})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_content}
        ]

    def _build_system_prompt(self, instruction: str) -> str:
        """
        通用 prompt - 包含通用交互原则
        """
        return f"""任务：{instruction}

## 屏幕结构（通用）
- 顶部区域：搜索栏、标题栏 (y 0-150)
- 内容区域：列表、视频、图片 (y 150-850)
- 底部区域：导航栏、标签栏、底部按钮 (y 850-1000)

## 坐标
坐标范围 [0, 1000]。左[0,0]，右[1000,1000]。x左→右，y上→下。

## 交互原则
1. 输入文字前先点击输入框/搜索框
2. 选择内容时点在内容区域(y>150)
3. 切换tab/导航时点在底部(y>850)
4. 打开应用时应用名与图标文字完全一致
5. 搜索结果页要点列表项，不是搜索栏

## 动作格式
CLICK:[[x, y]] - 点击坐标
TYPE:["文字"] - 输入文字
OPEN:["应用名"] - 打开应用
SCROLL:[[x1,y1],[x2,y2]] - 滚动
COMPLETE:[] - 任务完成

## 输出
仔细观察屏幕，根据任务目标决定下一步操作，用上述格式输出。"""

    def _parse_output(self, raw_output: str) -> tuple:
        """解析输出"""
        raw_output = raw_output.strip()

        # 函数调用格式
        func_patterns = {
            'CLICK': r'click\s*\(\s*point\s*=\s*"?\s*<point>\s*([\d.]+)\s+([\d.]+)\s*</point>\s*"?\s*\)',
            'TYPE': r'type\s*\(\s*content\s*=\s*["\']([^"\']+)["\']\s*\)',
            'OPEN': r'open\s*\(\s*app_name\s*=\s*["\']([^"\']+)["\']\s*\)',
            'COMPLETE': r'complete\s*\(\s*\)',
        }

        for action_name, pattern in func_patterns.items():
            match = re.search(pattern, raw_output, re.IGNORECASE)
            if match:
                return self._parse_funcall_params(action_name, match)

        # ACTION:[[...]] 格式
        for action in ['CLICK', 'TYPE', 'OPEN', 'COMPLETE']:
            if action == 'SCROLL':
                continue
            pattern = rf'{action}:\s*(\[[^\]]*\]|\S+)'
            match = re.search(pattern, raw_output.upper())
            if match:
                params = self._parse_params(action, match.group(1))
                if params is not None:
                    return action, params

        # SCROLL 格式
        scroll_match = re.search(r'SCROLL:\s*(\[\[[^\]]*\],\s*\[[^\]]*\]\])', raw_output.upper())
        if scroll_match:
            params = self._parse_params('SCROLL', scroll_match.group(1))
            if params:
                return 'SCROLL', params

        # 中文坐标
        point_match = re.search(r'<point>\s*([\d.]+)\s+([\d.]+)\s*</point>', raw_output)
        if point_match:
            x, y = int(float(point_match.group(1))), int(float(point_match.group(2)))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return 'CLICK', {"point": [x, y]}

        logger.warning(f"[Agent] Parse failed: {raw_output[:100]}")
        return "", {}

    def _parse_params(self, action: str, param_str: str) -> Optional[Dict[str, Any]]:
        """解析参数"""
        try:
            if action == 'CLICK':
                param_str = param_str.strip('[]')
                coords = json.loads(f"[{param_str}]")
                if len(coords) == 2:
                    return {"point": coords}

            elif action == 'TYPE':
                return {"text": param_str.strip("[]'\"")}

            elif action == 'OPEN':
                return {"app_name": param_str.strip("[]'\"")}

            elif action == 'SCROLL':
                data = json.loads(param_str)
                if isinstance(data, list) and len(data) == 2:
                    return {"start_point": data[0], "end_point": data[1]}

            elif action == 'COMPLETE':
                return {}

        except:
            pass
        return None

    def _parse_funcall_params(self, action: str, match: re.Match) -> Dict[str, Any]:
        """解析函数调用参数"""
        if action == 'CLICK':
            return {"point": [int(float(match.group(1))), int(float(match.group(2)))]}
        elif action == 'TYPE':
            return {"text": match.group(1)}
        elif action == 'OPEN':
            return {"app_name": match.group(1)}
        elif action == 'COMPLETE':
            return {}
        return {}

    def reset(self):
        """重置状态"""
        self._last_action = None
        self._last_params = None
