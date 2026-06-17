from dotenv import load_dotenv
load_dotenv()

import os
from serpapi import SerpApiClient
from typing import Any, Dict, Optional, Type, List, Callable, Union
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass, field
import inspect


class BaseToolInput(BaseModel):
    """
    工具输入的基类，所有工具输入都应继承此类。
    """
    pass


class BaseTool:
    """
    工具基类，所有工具都应继承此类。
    
    参考 LangChain 的 BaseTool 设计模式：
    - 使用 Pydantic 模型定义输入参数
    - 提供统一的执行接口
    - 自动生成工具描述
    """
    
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False
    
    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__.lower().replace("tool", "")
    
    def _run(self, **kwargs) -> Any:
        """
        同步执行工具的核心逻辑。
        子类必须实现此方法。
        """
        raise NotImplementedError("子类必须实现 _run 方法")
    
    def _arun(self, **kwargs) -> Any:
        """
        异步执行工具的核心逻辑（可选）。
        """
        raise NotImplementedError("异步执行未实现")
    
    def run(self, **kwargs) -> Any:
        """
        执行工具的入口方法。
        负责参数验证和调用 _run。
        """
        # 参数验证
        if self.args_schema:
            try:
                validated_input = self.args_schema(**kwargs)
                kwargs = validated_input.dict()
            except ValidationError as e:
                raise ValueError(f"参数验证失败: {e}")
        
        return self._run(**kwargs)
    
    async def arun(self, **kwargs) -> Any:
        """
        异步执行工具的入口方法。
        """
        if self.args_schema:
            try:
                validated_input = self.args_schema(**kwargs)
                kwargs = validated_input.dict()
            except ValidationError as e:
                raise ValueError(f"参数验证失败: {e}")
        
        return await self._arun(**kwargs)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        获取工具的结构化信息，用于 LLM 调用。
        返回格式符合 OpenAI Functions 规范。
        """
        parameters = {}
        
        if self.args_schema:
            # 兼容 Pydantic v2: 使用 model_fields 替代 __fields__
            model_fields = self.args_schema.model_fields
            # 从 Pydantic 模型提取参数信息
            for field_name, field_info in model_fields.items():
                param_type = field_info.annotation
                param_desc = field_info.description or ""
                
                # 转换类型为 JSON schema 类型
                json_type = "string"
                if param_type == int:
                    json_type = "integer"
                elif param_type == float:
                    json_type = "number"
                elif param_type == bool:
                    json_type = "boolean"
                elif param_type == list:
                    json_type = "array"
                elif param_type == dict:
                    json_type = "object"
                
                parameters[field_name] = {
                    "type": json_type,
                    "description": param_desc
                }
            
            # 确定必需参数
            required = [
                field_name for field_name, field_info in model_fields.items()
                if field_info.is_required()
            ]
        else:
            parameters = {"type": "object", "properties": {}}
            required = []
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                }
            }
        }
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


# ------------------------------
# 实战工具实现示例
# ------------------------------

class SearchToolInput(BaseToolInput):
    """搜索工具的输入参数模型"""
    query: str = Field(description="搜索关键词")
    location: str = Field(default="Guangzhou, Guangdong Province, China", 
                          description="搜索位置")
    language: str = Field(default="zh-cn", description="搜索语言")
    num_results: int = Field(default=3, description="返回结果数量")


class SearchTool(BaseTool):
    """
    基于 SerpApi 的网页搜索工具。
    
    用于搜索实时信息、事实性问题等。
    """
    name: str = "search"
    description: str = "网页搜索引擎。当你需要回答关于时事、事实以及在知识库中找不到的信息时，应使用此工具。"
    args_schema: Type[BaseModel] = SearchToolInput
    
    def _run(self, query: str, location: str = "Guangzhou, Guangdong Province, China", 
             language: str = "zh-cn", num_results: int = 3) -> str:
        """
        执行搜索。
        """
        print(f"🔍 正在执行 [SerpApi] 搜索: query={query}, location={location}")
        
        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"
            
            params = {
                "engine": "google",
                "q": query,
                "location": location,
                "api_key": api_key,
                "gl": "cn",
                "hl": language,
            }
            
            client = SerpApiClient(params)
            results = client.get_dict()
            
            # 优先返回直接答案
            if "answer_box_list" in results:
                return "\n".join(results["answer_box_list"])
            if "answer_box" in results and "answer" in results["answer_box"]:
                return results["answer_box"]["answer"]
            if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                return results["knowledge_graph"]["description"]
            if "organic_results" in results and results["organic_results"]:
                snippets = [
                    f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(results["organic_results"][:num_results])
                ]
                return "\n\n".join(snippets)
            
            return f"对不起，没有找到关于 '{query}' 的信息。"
        
        except Exception as e:
            return f"搜索时发生错误: {e}"


# ------------------------------
# 计算器工具
# ------------------------------

class CalculatorToolInput(BaseToolInput):
    """计算器工具的输入参数模型"""
    expression: str = Field(description="数学表达式，支持 +, -, *, /, //, %, **, () 等运算。例如: '(123 + 456) * 789 / 12'")


class CalculatorTool(BaseTool):
    """
    数学计算器工具。
    
    用于执行数学运算，支持加减乘除、幂运算、取余、括号等。
    当需要计算复杂数学表达式的结果时，应使用此工具。
    """
    name: str = "calculator"
    description: str = "数学计算器。当你需要计算数学表达式的结果时使用此工具。支持 +, -, *, /, //, %, ** 和括号。"
    args_schema: Type[BaseModel] = CalculatorToolInput
    
    def _run(self, expression: str) -> str:
        """
        安全地计算数学表达式。
        """
        print(f"🧮 正在计算: {expression}")
        
        # 允许的字符白名单
        import re as _re
        allowed_pattern = r'^[0-9+\-*/().%\s\*\*eE]+$'
        
        # 清理表达式：替换中文符号和常见别名
        cleaned = expression.strip()
        cleaned = cleaned.replace('×', '*').replace('÷', '/')
        cleaned = cleaned.replace('（', '(').replace('）', ')')
        cleaned = cleaned.replace('^', '**')
        
        if not _re.match(allowed_pattern, cleaned):
            return f"错误：表达式包含不允许的字符。仅支持数字和运算符 (+, -, *, /, //, %, **, ())，收到: {expression}"
        
        try:
            # 使用 compile + eval 限制只能访问数学运算
            code = compile(cleaned, '<calculator>', 'eval')
            # 禁止访问任何内置函数和变量
            result = eval(code, {"__builtins__": {}}, {})
            return str(result)
        except ZeroDivisionError:
            return "错误：除以零"
        except SyntaxError:
            return f"错误：表达式语法不正确: {expression}"
        except Exception as e:
            return f"计算错误: {e}"


# ------------------------------
# 工具执行器（ToolExecutor）
# ------------------------------

class ToolExecutor:
    """
    工具执行器，负责管理和调用工具。
    支持工具分类和关键词检索，适用于工具数量较多的场景。
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, str] = {}  # tool_name -> category
    
    def register_tool(self, tool: BaseTool, category: str = "通用") -> None:
        """
        注册一个工具实例。
        
        Args:
            tool: 工具实例
            category: 工具分类（如 "搜索", "计算", "文件" 等）
        """
        if tool.name in self.tools:
            raise ValueError(f"工具 '{tool.name}' 已存在")
        self.tools[tool.name] = tool
        self.tool_categories[tool.name] = category
        print(f"工具已注册: {tool.name} [{category}]")
    
    def register_function(self, func: Callable, name: Optional[str] = None, 
                          description: Optional[str] = None) -> None:
        """
        直接注册一个函数作为工具（自动包装）。
        
        Args:
            func: 函数对象
            name: 工具名称（默认使用函数名）
            description: 工具描述（默认使用函数文档字符串）
        """
        # 从函数签名创建 Pydantic 模型
        fields = {}
        sig = inspect.signature(func)
        
        for param_name, param in sig.parameters.items():
            # 获取类型注解
            field_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            
            # 获取默认值
            if param.default != inspect.Parameter.empty:
                default_value = param.default
                required = False
            else:
                default_value = ...  # Required
                required = True
            
            # 创建字段描述（从参数名推断）
            field_desc = f"{param_name} 参数"
            
            fields[param_name] = (field_type, Field(default=default_value, description=field_desc))
        
        # 动态创建 Pydantic 模型
        input_model = type(f"{func.__name__}Input", (BaseToolInput,), fields)
        
        # 创建工具类
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"{tool_name} 工具"
        
        class DynamicTool(BaseTool):
            name: str = tool_name
            description: str = tool_desc
            args_schema: Type[BaseModel] = input_model
            
            def _run(self, **kwargs):
                return func(**kwargs)
        
        self.register_tool(DynamicTool())
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        根据名称获取工具。
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        获取所有已注册工具的名称列表。
        """
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具的结构化信息。
        """
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_tool_info()
        return None
    
    def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的结构化信息列表。
        """
        return [tool.get_tool_info() for tool in self.tools.values()]
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        获取指定分类下的所有工具信息。
        """
        return [
            tool.get_tool_info() 
            for name, tool in self.tools.items() 
            if self.tool_categories.get(name) == category
        ]
    
    def list_categories(self) -> Dict[str, List[str]]:
        """
        获取所有分类及其包含的工具名称。
        返回: {"分类名": ["工具名1", "工具名2", ...]}
        """
        categories: Dict[str, List[str]] = {}
        for tool_name, category in self.tool_categories.items():
            categories.setdefault(category, []).append(tool_name)
        return categories
    
    def search_tools(self, keyword: str) -> List[Dict[str, Any]]:
        """
        根据关键词搜索工具（匹配工具名或描述）。
        """
        keyword_lower = keyword.lower()
        results = []
        for name, tool in self.tools.items():
            if keyword_lower in name.lower() or keyword_lower in tool.description.lower():
                results.append(tool.get_tool_info())
        return results
    
    def get_tools_summary(self) -> str:
        """
        获取工具的分类摘要，适用于工具数量较多时嵌入提示词。
        比完整 JSON Schema 更紧凑。
        """
        categories = self.list_categories()
        lines = []
        for category, tool_names in categories.items():
            lines.append(f"[{category}]")
            for name in tool_names:
                tool = self.tools[name]
                lines.append(f"  - {name}: {tool.description}")
        return "\n".join(lines)
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行指定的工具。
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
        
        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 未找到")
        
        return tool.run(**kwargs)


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == '__main__':
    # 初始化工具执行器
    executor = ToolExecutor()
    
    # 注册搜索工具
    executor.register_tool(SearchTool())
    
    # 注册一个简单的函数工具
    def greet(name: str, greeting: str = "你好") -> str:
        """打招呼工具"""
        return f"{greeting}，{name}！"
    
    executor.register_function(greet)
    
    # 列出所有工具
    print("=== 已注册工具 ===")
    for tool_name in executor.list_tools():
        print(f"- {tool_name}")
    
    # 获取工具信息（用于 LLM 调用）
    print("\n=== 搜索工具信息 ===")
    print(executor.get_tool_info("search"))
    
    # 执行工具
    print("\n=== 执行搜索工具 ===")
    result = executor.execute("search", query="英伟达最新GPU")
    print(result)
    
    # 执行函数工具
    print("\n=== 执行打招呼工具 ===")
    result = executor.execute("greet", name="张三", greeting="欢迎")
    print(result)
