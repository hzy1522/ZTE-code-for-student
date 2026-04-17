"""
Agent 派生类实现

选手需要实现自己的 Agent，继承 BaseAgent 并实现 act 方法。

改进版本：
1. 动态 Task-Type 检测 + 模块化 Prompt
2. 结构化 JSON 输出替代复杂正则
3. 历史模式分析 + 自我校正
4. 增强的历史上下文
"""

import re
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from agent_base import (
    BaseAgent, AgentInput, AgentOutput
)

logger = logging.getLogger(__name__)


# ============ Task Type Detection ============

class TaskType(Enum):
    SEARCH = "search"           # 搜索类任务
    NAVIGATION = "navigation"   # 导航/tab切换
    CONTENT_INTERACT = "content"  # 内容交互（播放/收藏/点赞）
    SHOPPING = "shopping"       # 外卖/电商
    TRANSPORT = "transport"     # 打车/交通
    MIXED = "mixed"             # 混合任务


class TaskDetector:
    """从 instruction 中检测任务类型"""

    TASK_PATTERNS = {
        TaskType.SEARCH: [
            r'搜索', r'查找', r'搜一下', r'搜索.*?筛选', r'在.*?里搜索',
        ],
        TaskType.NAVIGATION: [
            r'进入', r'打开.*?tab', r'切换到', r'去.*?页面', r'点.*?tab',
        ],
        TaskType.CONTENT_INTERACT: [
            r'播放', r'收藏', r'点赞', r'关注', r'分享', r'下载',
        ],
        TaskType.SHOPPING: [
            r'外卖', r'购物', r'添加.*?购物车', r'点外卖', r'买', r'下单',
        ],
        TaskType.TRANSPORT: [
            r'打车', r'从.*?去.*?', r'导航到', r'路线', r'呼叫',
        ],
    }

    APP_PATTERNS = {
        '芒果TV': TaskType.CONTENT_INTERACT,
        '抖音': TaskType.CONTENT_INTERACT,
        '快手': TaskType.CONTENT_INTERACT,
        '哔哩哔哩': TaskType.CONTENT_INTERACT,
        '腾讯视频': TaskType.CONTENT_INTERACT,
        '爱奇艺': TaskType.CONTENT_INTERACT,
        '美团': TaskType.SHOPPING,
        '大众点评': TaskType.SHOPPING,
        '京东': TaskType.SHOPPING,
        '淘宝': TaskType.SHOPPING,
        '拼多多': TaskType.SHOPPING,
        '百度地图': TaskType.TRANSPORT,
        '高德地图': TaskType.TRANSPORT,
        '滴滴': TaskType.TRANSPORT,
        '去哪旅行': TaskType.SHOPPING,
        '喜马拉雅': TaskType.CONTENT_INTERACT,
    }

    def detect(self, instruction: str) -> TaskType:
        instruction_lower = instruction.lower()

        # First check app-specific patterns
        for app, task_type in self.APP_PATTERNS.items():
            if app in instruction:
                return task_type

        # Then check general patterns
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    return task_type

        return TaskType.MIXED


# ============ Task-Specific Rules ============

TASK_SPECIFIC_RULES = {
    TaskType.SEARCH: """
## 搜索类任务规则
【关键流程】
1. 点击搜索入口（放大镜图标，通常在顶部右侧）
2. 点击输入框区域激活
3. 输入搜索词
4. 如果出现候选列表，点击列表中的目标项
5. 如果没有候选列表，点击键盘"搜索"键或屏幕确认按钮

【禁止】
- 输入搜索词后、执行搜索前点击任何内容区元素
- 搜索结果已显示后不要点击内容区，直接判断任务是否完成

【何时完成】
- 筛选类：筛选条件已设置即完成
- 搜索类：搜索结果已显示即完成
""",
    TaskType.NAVIGATION: """
## 导航/切换类任务规则
1. 底部tab栏位于y>850区域，点击目标tab
2. 切换tab后等待页面更新，确认当前页面
3. 如果页面没有变化，可能是点击位置不对，重新点击tab区域
""",
    TaskType.CONTENT_INTERACT: """
## 内容交互类任务规则
【关键流程】
1. 播放类：点击播放按钮或视频区域，等待播放开始
2. 收藏/点赞/关注/分享类：
   - 必须执行该操作，不能仅完成前置步骤就结束
   - 找到对应按钮并点击
3. 芒果TV底部tab切换：点击目标tab后确认页面变化

【何时完成】
- 播放类：视频开始播放即完成
- 收藏/点赞/关注/分享类：操作执行完毕即完成
- 【禁止】收藏/点赞等操作仅完成前置步骤（如搜索、进入页面）就结束
""",
    TaskType.SHOPPING: """
## 电商/外卖类任务规则
【关键流程】必须严格按顺序执行：
1. 点击顶部搜索框（y<150区域）
2. 点击搜索框内部的输入区域（y<850内容区）
3. 输入搜索内容
4. 如果出现下拉列表，【必须】点击列表中的目标项，不要继续输入
5. 进入店铺/搜索结果后，找到目标商品，点击添加
6. 【重要】添加完成后必须【立即】点击结算按钮（y>850），禁止继续操作

【禁止】
- 输入文字前不点击输入框
- 出现下拉列表时不点击列表项而继续输入
- 添加商品后不立即结算，继续浏览
""",
    TaskType.TRANSPORT: """
## 打车/交通类任务规则
【关键流程】
1. 百度地图打车界面有上下两个输入框：【上方是出发地，下方是目的地】
2. 先点击上方输入框，输入或选择出发地
3. 再点击下方输入框，输入或选择目的地
4. 确认起点终点后，寻找并点击"确认打车"或"叫车"按钮

【禁止】
- 不要默认当前位置，必须按照任务指定的地点操作
- 不要先输入目的地再输入出发地，必须按顺序

【何时完成】
- 等待司机接单后任务完成，或点击取消后完成
""",
}

CORE_RULES = """
## 坐标规范
- 归一化坐标范围 [0, 1000]，只输出整数
- 顶部 (y 0-150)：搜索栏、标题栏
- 内容区 (y 150-850)：列表、视频、图片、商品
- 底部 (y 850-1000)：导航栏、结算按钮、确认按钮

## 坐标预测技巧
- 按钮和图标通常在区域中心位置
- 如果不确定元素精确位置，优先点击区域中心
- 搜索框宽度较大，中心点约为 (500, 50-100)
- 列表项较窄，宽度约 (0-400)，中心约 x=200
- 右侧操作按钮通常在 x > 700 区域
- 底部按钮通常 y > 850

## 决策检查清单
在输出坐标前，检查：
1. 这个坐标在屏幕范围内吗？(0 <= x <= 1000, 0 <= y <= 1000)
2. 这个坐标在正确的区域内吗？(搜索框在顶部，结算在底部)
3. 这个坐标与屏幕上看到的元素对齐吗？
4. 上一次类似操作成功了吗？(如果失败，换一个区域尝试)

## 核心交互原则（必须遵守）
1. 【强制】输入文字前必须先点击输入框/搜索框
2. 【强制】出现下拉列表/候选列表时，必须点击列表项，不能继续输入
3. 点击内容时必须在图标/文字/按钮上，不能点在空白处
4. 点击列表项时点在文字或图标上，不要点在行间空白
5. 打开应用时应用名与图标文字完全一致

## 任务完成标准
- 搜索/筛选类：结果已显示即完成，禁止再点击任何元素
- 收藏/点赞/关注/分享类：必须执行该操作才能结束
- 输入后出现候选列表：必须点击列表第一项
- 外卖/购物类：添加完成后立即点结算按钮
"""

# ============ Region Hotspot Hints ============

HOTSPOT_HINTS = {
    TaskType.SEARCH: """
## 常见元素位置提示
- 搜索入口/放大镜图标：通常在屏幕右上角 (x: 700-950, y: 30-120)
- 搜索框内输入区域：屏幕顶部中央 (x: 200-800, y: 40-120)
- 搜索按钮：搜索框右侧或屏幕右上角
- 返回按钮：屏幕左上角 (x: 0-100, y: 30-80)
""",
    TaskType.NAVIGATION: """
## 常见元素位置提示
- 底部Tab栏：y > 850 区域
- 返回按钮：屏幕左上角 (x: 0-100, y: 30-80)
- 页面标题：屏幕顶部中央 (x: 200-800, y: 30-80)
""",
    TaskType.CONTENT_INTERACT: """
## 常见元素位置提示
- 播放按钮：视频中央或底部 (x: 400-600, y: 400-600 或 y > 800)
- 收藏/点赞/关注按钮：通常在屏幕右侧 (x > 700) 或视频下方
- 收藏图标：心形或星形，通常在视频/文章右侧
- 返回按钮：屏幕左上角 (x: 0-100, y: 30-80)
- 下载按钮：通常在屏幕右下角或列表项右侧
""",
    TaskType.SHOPPING: """
## 常见元素位置提示
- 顶部搜索框：y < 150 区域，屏幕中央到右侧
- 搜索框内输入区域：点击搜索框后出现
- 下拉列表项：y 100-400 区域
- 店铺入口：列表中通常在左侧
- 商品添加按钮：通常在商品右侧或底部 (+、添加 等)
- 结算/提交订单按钮：y > 850，屏幕底部中央或右侧
- 地址选择：屏幕中部列表
""",
    TaskType.TRANSPORT: """
## 常见元素位置提示
- 百度地图打车输入框：上方是出发地 (y: 200-400)，下方是目的地 (y: 400-600)
- 出发地输入框：屏幕上半部分
- 目的地输入框：屏幕中半部分（在出发地下方）
- 确认/叫车按钮：屏幕底部 (y > 800)
- 选择列表项：屏幕中部 (y: 200-600)
- 返回按钮：屏幕左上角 (x: 0-100, y: 30-80)
""",
}


# ============ Output Parser ============

class OutputParser:
    """结构化输出解析器 - JSON 优先，正则回退"""

    def parse(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        raw_output = raw_output.strip()

        # Strategy 1: Try JSON
        action, params = self._try_json(raw_output)
        if action:
            return action, params

        # Strategy 2: Structured format CLICK:[[x, y]]
        action, params = self._try_structured(raw_output)
        if action:
            return action, params

        # Strategy 3: Simplified regex
        action, params = self._try_regex(raw_output)
        if action:
            return action, params

        # Strategy 4: Chinese keywords (last resort)
        action, params = self._try_chinese(raw_output)
        if action:
            return action, params

        logger.warning(f"[Agent] Parse failed: {raw_output[:100]}")
        return "", {}

    def _try_json(self, raw: str) -> Tuple[Optional[str], Optional[Dict]]:
        """尝试 JSON 解析"""
        try:
            # 尝试从 markdown 代码块中提取
            json_match = re.search(r'```(?:json)?\s*(\{[^}]+\})\s*```', raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试找包含 action 字段的 JSON
                json_match = re.search(r'\{[^}]*"action"[^}]*\}', raw, re.DOTALL)
                if not json_match:
                    return None, None
                json_str = json_match.group(0)

            data = json.loads(json_str)
            action = data.get('action', '').upper()

            if action not in {'CLICK', 'TYPE', 'SCROLL', 'OPEN', 'COMPLETE'}:
                return None, None

            params = data.get('params', {})

            if action == 'CLICK':
                point = params.get('point', [])
                if len(point) == 2:
                    return action, {'point': [int(point[0]), int(point[1])]}
            elif action == 'TYPE':
                return action, {'text': params.get('text', '')}
            elif action == 'OPEN':
                return action, {'app_name': params.get('app_name', '')}
            elif action == 'SCROLL':
                start = params.get('start_point', [])
                end = params.get('end_point', [])
                if len(start) == 2 and len(end) == 2:
                    return action, {
                        'start_point': [int(start[0]), int(start[1])],
                        'end_point': [int(end[0]), int(end[1])]
                    }
            elif action == 'COMPLETE':
                return action, {}

            return None, None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None, None

    def _try_structured(self, raw: str) -> Tuple[Optional[str], Optional[Dict]]:
        """解析 CLICK:[[x, y]] 等格式"""
        raw_upper = raw.upper()

        # CLICK:[[x, y]]
        match = re.search(r'CLICK:\s*\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]', raw_upper)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return 'CLICK', {'point': [x, y]}

        # TYPE:["text"]
        match = re.search(r'TYPE:\s*\[\s*"([^"]+)"\s*\]', raw)
        if match:
            return 'TYPE', {'text': match.group(1)}

        # OPEN:["app"]
        match = re.search(r'OPEN:\s*\[\s*"([^"]+)"\s*\]', raw)
        if match:
            return 'OPEN', {'app_name': match.group(1)}

        # COMPLETE:[]
        if re.search(r'COMPLETE:\s*\[\s*\]', raw_upper):
            return 'COMPLETE', {}

        return None, None

    def _try_regex(self, raw: str) -> Tuple[Optional[str], Optional[Dict]]:
        """简化的正则解析"""
        raw_upper = raw.upper()

        # 函数调用格式
        func_patterns = {
            'CLICK': r'click\s*\(\s*point\s*=\s*"?\s*<point>\s*(\d+)\s+(\d+)\s*</point>\s*"?\s*\)',
            'TYPE': r'type\s*\(\s*content\s*=\s*["\']([^"\']+)["\']\s*\)',
            'OPEN': r'open\s*\(\s*app_name\s*=\s*["\']([^"\']+)["\']\s*\)',
            'COMPLETE': r'complete\s*\(\s*\)',
        }

        for action, pattern in func_patterns.items():
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                if action == 'CLICK':
                    return 'CLICK', {'point': [int(match.group(1)), int(match.group(2))]}
                elif action == 'TYPE':
                    return 'TYPE', {'text': match.group(1)}
                elif action == 'OPEN':
                    return 'OPEN', {'app_name': match.group(1)}
                elif action == 'COMPLETE':
                    return 'COMPLETE', {}

        # 裸坐标 [x, y] 或 x y
        bare = re.search(r'\[?\s*(\d{2,4})\s*,\s*(\d{2,4})\s*\]?', raw)
        if bare:
            x, y = int(bare.group(1)), int(bare.group(2))
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return 'CLICK', {'point': [x, y]}

        return None, None

    def _try_chinese(self, raw: str) -> Tuple[Optional[str], Optional[Dict]]:
        """中文关键词匹配（最后回退）"""
        if re.search(r'完成|结束了', raw):
            return 'COMPLETE', {}

        type_match = re.search(r'输入.*?["\']([^"\']+)["\']', raw)
        if type_match:
            return 'TYPE', {'text': type_match.group(1)}

        open_match = re.search(r'打开.*?["\']([^"\']+)["\']', raw)
        if open_match:
            return 'OPEN', {'app_name': open_match.group(1)}

        return None, None


# ============ History Analyzer ============

class HistoryAnalysis:
    """历史动作分析结果"""
    def __init__(self):
        self.repeated_clicks = []  # 相似位置重复点击
        self.failed_actions = []   # 失败的动作
        self.warnings = []       # 警告消息
        self.confidence = 1.0     # 置信度


class ActionHistoryAnalyzer:
    """分析历史动作，检测失败模式"""

    COORD_THRESHOLD = 80  # 距离阈值

    def analyze(self, history_actions: List[Dict]) -> HistoryAnalysis:
        if not history_actions:
            return HistoryAnalysis()

        analysis = HistoryAnalysis()

        # 收集 CLICK 动作
        clicks = []
        for h in history_actions:
            if h.get('action') == 'CLICK':
                point = h.get('parameters', {}).get('point', [])
                if len(point) == 2:
                    clicks.append({
                        'step': h.get('step'),
                        'point': point,
                        'is_valid': h.get('is_valid', True)
                    })

        # 检测相似位置重复失败
        for i, click in enumerate(clicks):
            if not click['is_valid']:
                for j, other in enumerate(clicks[i+1:], start=i+1):
                    if not other['is_valid']:
                        dist = math.sqrt(
                            (click['point'][0] - other['point'][0])**2 +
                            (click['point'][1] - other['point'][1])**2
                        )
                        if dist < self.COORD_THRESHOLD:
                            analysis.repeated_clicks.append({
                                'points': [click['point'], other['point']],
                                'steps': [click['step'], other['step']]
                            })

        # 收集失败动作
        for h in history_actions:
            if not h.get('is_valid', True):
                analysis.failed_actions.append(h)

        # 生成警告
        analysis.warnings = self._generate_warnings(analysis, len(history_actions))

        # 计算置信度
        analysis.confidence = max(0.3, 1.0 - len(analysis.failed_actions) * 0.1)

        return analysis

    def _generate_warnings(self, analysis: HistoryAnalysis, total_steps: int) -> List[str]:
        warnings = []

        if len(analysis.repeated_clicks) >= 1:
            warnings.append(
                "检测到点击相似位置但未成功，请重新分析目标元素位置"
            )

        if len(analysis.failed_actions) >= 3:
            warnings.append(
                "近期有多次失败动作，请仔细观察当前页面"
            )

        # 检测连续点击模式
        recent_actions = [h.get('action', '') for h in analysis.failed_actions[-4:]]
        if recent_actions.count('CLICK') >= 3:
            warnings.append(
                "连续多次点击未成功，请确认：1) 点击位置是否正确？2) 是否需要先触发其他操作？"
            )

        return warnings


# ============ Main Agent Class ============

class Agent(BaseAgent):
    """
    GUI Agent 实现类 - 改进版本

    改进点：
    1. 动态 Task-Type 检测 + 模块化 Prompt
    2. 结构化 JSON 输出解析
    3. 历史模式分析 + 自我校正
    4. 增强的历史上下文
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._last_action = None
        self._last_params = None
        self._task_detector = TaskDetector()
        self._output_parser = OutputParser()
        self._history_analyzer = ActionHistoryAnalyzer()

    def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        return super()._call_api(messages, temperature=0)

    def act(self, input_data: AgentInput) -> AgentOutput:
        messages = self.generate_messages(input_data)

        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content
            usage = self.extract_usage_info(response)
            action, parameters = self._parse_output(raw_output)

            self._last_action = action
            self._last_params = parameters

            logger.info(f"[Agent Raw Output] {raw_output[:200]}")
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
        task_type = self._task_detector.detect(input_data.instruction)
        system_prompt = self._build_adaptive_prompt(
            input_data.instruction,
            task_type,
            input_data.step_count
        )

        history_analysis = self._history_analyzer.analyze(input_data.history_actions)

        current_content = [
            {"type": "text", "text": f"任务: {input_data.instruction}"},
            {"type": "text", "text": f"第{input_data.step_count}步"},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(input_data.current_image)}
            }
        ]

        # 添加历史上下文
        if input_data.history_actions:
            history_text = self._format_history_context(
                input_data.history_actions[-8:],
                history_analysis
            )
            current_content.append({"type": "text", "text": history_text})

        # 添加警告
        if history_analysis.warnings:
            current_content.append({
                "type": "text",
                "text": "【注意】" + "；".join(history_analysis.warnings)
            })

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_content}
        ]

    def _build_adaptive_prompt(self, instruction: str, task_type: TaskType, step_count: int) -> str:
        parts = [
            f"任务：{instruction}",
            CORE_RULES,
        ]

        # 添加任务特定规则
        task_rules = TASK_SPECIFIC_RULES.get(task_type, "")
        if task_rules:
            parts.append(task_rules)

        # 添加区域热点提示
        hotspot_hint = HOTSPOT_HINTS.get(task_type, "")
        if hotspot_hint:
            parts.append(hotspot_hint)

        # 添加步骤指导
        if step_count == 1:
            parts.append("\n## 第一步\n仔细观察截图，确定任务入口：\n- 如果是打开应用：找应用图标\n- 如果是搜索：找放大镜图标或搜索框\n- 如果是内容操作：找对应的按钮或内容")
        elif step_count <= 3:
            parts.append("\n## 早期步骤\n关键操作阶段：\n- 按照任务类型规则执行操作\n- 注意是否出现下拉列表/候选列表，如有必须点击列表项\n- 不要遗漏必要步骤")
        else:
            parts.append("\n## 后期步骤\n收尾阶段：\n- 检查任务是否完成\n- 如果是购物/外卖：是否已点击结算\n- 如果是收藏/点赞：是否已执行操作\n- 如果是搜索：结果是否已显示")

        # 输出格式
        parts.append("""
## 输出格式
直接输出决策，不要包含其他内容：

CLICK:[[x, y]]
TYPE:["文字"]
OPEN:["应用名"]
COMPLETE:[]""")

        return "\n".join(parts)

    def _format_history_context(self, recent_actions: List[Dict], analysis: HistoryAnalysis) -> str:
        if not recent_actions:
            return ""

        parts = ["## 最近操作"]
        for h in recent_actions:
            step = h.get('step', '?')
            action = h.get('action', '')
            params = h.get('parameters', {})
            is_valid = h.get('is_valid', True)
            status = "成功" if is_valid else "失败"

            if action == 'CLICK' and 'point' in params:
                parts.append(f"  步骤{step}: 点击{params['point']} [{status}]")
            elif action == 'TYPE' and 'text' in params:
                text = params['text'][:10] + "..." if len(params['text']) > 10 else params['text']
                parts.append(f"  步骤{step}: 输入\"{text}\" [{status}]")
            elif action == 'OPEN' and 'app_name' in params:
                parts.append(f"  步骤{step}: 打开{params['app_name']} [{status}]")
            elif action == 'COMPLETE':
                parts.append(f"  步骤{step}: 完成 [{status}]")
            else:
                parts.append(f"  步骤{step}: {action} [{status}]")

        return "\n".join(parts)

    def _parse_output(self, raw_output: str) -> tuple:
        return self._output_parser.parse(raw_output)

    def reset(self):
        self._last_action = None
        self._last_params = None