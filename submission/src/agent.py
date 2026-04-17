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
        """调用 API - 使用父类的安全实现，temperature=0 最确定性"""
        return super()._call_api(messages, temperature=0)

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
        通用 prompt - 包含通用交互原则和任务完成判断
        """
        return f"""任务：{instruction}

## 屏幕结构
- 顶部 (y 0-150)：搜索栏、标题栏
- 内容 (y 150-850)：列表、视频、图片
- 底部 (y 850-1000)：导航栏、按钮

## 坐标
坐标范围 [0, 1000]，只输出整数。

## 交互原则
1. 【强制】输入文字前必须先点击输入框/搜索框（即使认为不需要也要先点击）
2. 选择内容时点在内容区域(y>150)，且必须点在图标/文字/按钮上，不能点在空白处
3. 切换tab/导航时点在底部(y>850)
4. 打开应用时应用名与图标文字完全一致
5. 搜索结果页要点列表项，不是搜索栏
6. 点击列表项时要点在文字或图标上，不要点在行与行之间的空白区域

## 任务完成判断
当用户任务的所有要求都已执行完成时，输出 COMPLETE:[]。
【重要】输入文字后，如果屏幕上有"发布"、"发送"、"提交"、"确认"等按钮，必须点击完成后才能 COMPLETE。
【重要】如果任务要求已完成（如搜索结果已显示、筛选已设置、播放已开始），不要再点击任何元素，直接 COMPLETE。
【重要】如果任务要求"收藏"、"点赞"、"关注"、"分享"等操作，必须执行该操作后才能 COMPLETE，不能仅完成前置步骤（如搜索）就结束。

## 动作格式
CLICK:[[x, y]]
TYPE:["文字"]
OPEN:["应用名"]
COMPLETE:[]"""

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
            if action == 'CLICK':
                pattern = rf'{action}:\s*(\[\[[^\]]+\]\])'
            elif action == 'SCROLL':
                pattern = rf'{action}:\s*(\[\[[^\]]+\],\s*\[[^\]]+\]\])'
            else:
                pattern = rf'{action}:\s*(\[[^\]]*\])'
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

        # 中文坐标 <point>x y</point>
        point_match = re.search(r'<point>\s*([\d.]+)\s+([\d.]+)\s*</point>', raw_output)
        if point_match:
            x, y = int(float(point_match.group(1))), int(float(point_match.group(2)))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return 'CLICK', {"point": [x, y]}

        # 裸坐标格式 CLICK [x, y] 或 click [x y]
        bare_point = re.search(r'\[?\s*([\d]{2,4})\s*,\s*([\d]{2,4})\s*\]?', raw_output, re.IGNORECASE)
        if bare_point:
            x, y = int(bare_point.group(1)), int(bare_point.group(2))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return 'CLICK', {"point": [x, y]}

        # 中文动作关键词
        if re.search(r'完成|结束了', raw_output):
            return 'COMPLETE', {}
        if re.search(r'输入|打字|填', raw_output):
            text_match = re.search(r'["\"\']([^"\']+)["\'"]', raw_output)
            if text_match:
                return 'TYPE', {"text": text_match.group(1)}
        if re.search(r'打开|启动|应用', raw_output):
            app_match = re.search(r'["\"\']([^"\']+)["\'"]', raw_output)
            if app_match:
                return 'OPEN', {"app_name": app_match.group(1)}

        logger.warning(f"[Agent] Parse failed: {raw_output[:100]}")
        return "", {}

    def _parse_params(self, action: str, param_str: str) -> Optional[Dict[str, Any]]:
        """解析参数"""
        try:
            if action == 'CLICK':
                param_str = param_str.strip('[]')
                # 支持空格分隔和逗号分隔的坐标，先分割再清理
                parts = [p.strip() for p in param_str.replace(' ', ',').split(',')]
                coords = [int(p) for p in parts if p]
                if len(coords) == 2:
                    return {"point": coords}

            elif action == 'TYPE':
                return {"text": param_str.strip("[]'\"")}

            elif action == 'OPEN':
                return {"app_name": param_str.strip("[]'\"")}

            elif action == 'SCROLL':
                # 支持空格分隔的坐标
                param_str_normalized = param_str.replace(' ', ',')
                data = json.loads(param_str_normalized)
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
