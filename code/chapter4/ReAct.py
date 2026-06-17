import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, SearchTool, CalculatorTool
import json


# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
你是一个可以调用工具的智能助手。

可用工具：
{tools}

你需要通过多步思考与行动来解决用户问题。
每一步你都必须严格按照以下格式输出，只能包含一个 Thought 和一个 Action：

Thought: 你的思考过程，分析当前情况并决定下一步。
Action: 你要执行的操作，必须是以下二者之一：
- 调用工具：输出一个JSON对象，格式为 {{"tool": "工具名", "params": {{"参数名": "值"}}}}
- 结束任务：{{"tool": "Finish", "params": {{"answer": "最终答案"}}}}

之后你会收到一个 Observation，里面是工具返回的结果。请基于它继续下一轮思考，直到得出最终答案。
{failure_hint}
当前对话历史：
{history}

用户问题：{question}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        self.consecutive_failures = 0  # 连续失败计数
        self.max_consecutive_failures = 3  # 超过此阈值时触发引导

    def _build_failure_hint(self) -> str:
        """根据连续失败次数生成引导提示。"""
        if self.consecutive_failures < self.max_consecutive_failures:
            return ""
        
        # 失败次数超过阈值，提供强引导
        tools_summary = self.tool_executor.get_tools_summary()
        hint = (
            f"\n\u26a0\ufe0f 注意：你已经连续 {self.consecutive_failures} 次工具调用失败。"
            f"请仔细检查以下可用工具列表，确保使用正确的工具名和参数格式：\n"
            f"{tools_summary}\n"
            f"请重新审视你的策略，选择一个合适的工具，或者如果你已有足够信息，请使用 Finish 输出答案。\n"
        )
        return hint

    def run(self, question: str):
        self.history = []
        self.consecutive_failures = 0
        cur_step = 0
        
        while cur_step < self.max_steps:
            cur_step += 1
            print(f"\n————第{cur_step}步————")
            
            tools_desc = json.dumps(self.tool_executor.get_all_tools_info(), ensure_ascii=False, indent=2)
            history = "\n".join(self.history)
            failure_hint = self._build_failure_hint()
            
            #格式化提示词
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history,
                failure_hint=failure_hint
            )
            
            message = {"role": "user", "content": prompt}
            response = self.llm_client.think(messages=[message])
            if not response:
                print("错误：LLM未能返回有效响应。")
                break
            
            thought, action = self._parse_output(response)
            self.history.append(f"Thought: {thought or '(无)'}")
            if not action:
                print("错误：LLM未返回有效动作。")
                self.history.append("Observation: 未按照要求返回Action，请严格按照格式输出。")
                self.consecutive_failures += 1
                continue
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                self.history.append(f"Action: {action}")
                self.history.append("Observation: 无效的Action JSON格式，请检查。")
                self.consecutive_failures += 1
                continue

            if tool_name == "Finish":
                final_answer = tool_input.get("answer", "") if isinstance(tool_input, dict) else str(tool_input)
                print(f"\n\u2705 Finish: {final_answer}")
                return final_answer

            # 检查工具是否存在
            if not self.tool_executor.get_tool(tool_name):
                available = self.tool_executor.list_tools()
                error_msg = f"工具 '{tool_name}' 不存在。可用工具: {', '.join(available)}"
                print(f"\u274c {error_msg}")
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {error_msg}")
                self.consecutive_failures += 1
                continue

            print(f"\U0001f527 Action: {tool_name}, 参数: {tool_input}")
            try:
                tool_output = self.tool_executor.execute(tool_name, **tool_input)
                self.consecutive_failures = 0  # 成功执行，重置失败计数
            except Exception as e:
                tool_output = f"工具执行出错: {e}"
                self.consecutive_failures += 1

            print(f"\U0001f4cb Observation: {tool_output}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {tool_output}")
        
        print("\n\u26a0\ufe0f 已达到最大步数限制，未能得出最终答案。")
        return None
            
    def _parse_output(self, response: str):
        thought = re.search(r"\s*Thought:\s*(.*?)(?=Action:|$)", response, re.DOTALL)
        action = re.search(r"\s*Action:\s*(.*?)$", response, re.DOTALL)
        thought = thought.group(1).strip() if thought else None
        action = action.group(1).strip() if action else None

        # 兜底：如果没有 Action: 前缀，尝试从原始响应中提取 JSON
        if not action:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "tool" in data:
                        action = json_match.group(0).strip()
                except json.JSONDecodeError:
                    pass

        return thought, action
    
    def _parse_action(self, action_text):
        json_match = re.search(r"\s*(\{.*\})", action_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data["tool"], data["params"]
            except (json.JSONDecodeError, KeyError):
                pass
        return None, None
        
if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    tool_executor.register_tool(SearchTool(), category="搜索")
    tool_executor.register_tool(CalculatorTool(), category="计算")
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "计算 (123 + 456) × 789 / 12 的结果"
    agent.run(question)
