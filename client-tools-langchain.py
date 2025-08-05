import subprocess
import sys
import argparse
import asyncio
import json
import os
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack

# LangChain相关导入 - 使用新版本导入方式
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("请安装 langchain-openai: pip install langchain-openai")
    from langchain_community.chat_models import ChatOpenAI

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException

# 简化内存管理，不使用已弃用的ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# MCP 客户端相关导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 环境变量加载相关
from dotenv import load_dotenv

# Pydantic用于数据验证
from pydantic import BaseModel, Field

class InMemoryChatHistory(BaseChatMessageHistory):
    """简单的内存聊天历史实现"""
    
    def __init__(self):
        self._messages = []
    
    @property
    def messages(self):
        return self._messages
    
    def add_message(self, message):
        self._messages.append(message)
    
    def clear(self):
        self._messages.clear()

def run_python_file(interpreter_path, script_path, script_args=None):
    """运行Python文件的工具函数"""
    command = [interpreter_path, script_path]
    if script_args:
        command.extend(script_args)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
        
        return result.returncode
    
    except subprocess.CalledProcessError as e:
        print(f"错误: 脚本返回非零退出状态 {e.returncode}")
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return e.returncode
    
    except FileNotFoundError:
        print(f"错误: 指定的Python解释器 '{interpreter_path}' 未找到")
        return 1
    
    except Exception as e:
        print(f"错误: 执行脚本时发生意外错误: {e}")
        return 1

class MCPToolInput(BaseModel):
    """MCP工具输入参数的基础模型"""
    pass

class MCPTool(BaseTool):
    """MCP工具的LangChain包装器"""
    
    def __init__(self, name: str, description: str, input_schema: dict, 
                 client_session: ClientSession, server_id: str):
        # 动态创建输入参数模型
        class DynamicInput(BaseModel):
            pass
        
        # 根据input_schema添加字段
        if "properties" in input_schema:
            for prop_name, prop_info in input_schema["properties"].items():
                field_type = str  # 简化处理，都当作字符串
                default_value = prop_info.get("default", ...)
                description = prop_info.get("description", "")
                
                DynamicInput.__annotations__[prop_name] = field_type
                setattr(DynamicInput, prop_name, Field(default=default_value, description=description))
        
        super().__init__(
            name=name,
            description=description,
            args_schema=DynamicInput
        )
        self.client_session = client_session
        self.server_id = server_id
        self.input_schema = input_schema
    
    async def _arun(self, **kwargs) -> str:
        """异步运行MCP工具"""
        try:
            print(f"调用MCP工具: {self.name}, 参数: {kwargs}")
            result = await self.client_session.call_tool(self.name, kwargs)
            
            # 处理结果
            tool_result_content = result.content
            if isinstance(tool_result_content, list):
                text_content = ""
                for item in tool_result_content:
                    if hasattr(item, 'text'):
                        text_content += item.text
                tool_result_content = text_content
            elif not isinstance(tool_result_content, str):
                tool_result_content = str(tool_result_content)
            
            print(f"工具执行结果: {tool_result_content}")
            return tool_result_content
            
        except Exception as e:
            print(f"工具执行错误: {e}")
            return f"工具执行失败: {str(e)}"
    
    def _run(self, **kwargs) -> str:
        """同步运行（通过异步实现）"""
        # 在LangChain中，通常需要同步接口
        # 这里需要获取当前事件循环或创建新的
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在异步环境中，创建任务
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(**kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
        except:
            # 创建新的事件循环
            return asyncio.run(self._arun(**kwargs))

class MCPCallback(BaseCallbackHandler):
    """MCP客户端的回调处理器"""
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        print(f"开始执行工具: {serialized.get('name', 'unknown')}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"工具执行完成")
    
    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        print(f"工具执行错误: {error}")

class LangChainMCPClient:
    """基于LangChain的MCP客户端"""
    
    def __init__(self):
        # 先初始化基本属性
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        self.tools_map = {}
        self.mcp_tools = []
        
        # 加载环境变量（改进错误处理）
        try:
            load_dotenv()
        except Exception as e:
            print(f"加载环境变量时出错: {e}")
            print("请检查 .env 文件格式")
        
        # 获取环境变量，提供默认值
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL") 
        model_name = os.getenv("MODEL", "deepseek-chat")
        
        if not api_key:
            print("警告: 未找到 API_KEY 环境变量")
        if not base_url:
            print("警告: 未找到 BASE_URL 环境变量")
        
        # LangChain相关
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                temperature=0.5,
                max_tokens=4096
            )
        except Exception as e:
            print(f"初始化 ChatOpenAI 时出错: {e}")
            raise
        
        # 读取系统提示词
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        except FileNotFoundError:
            print("警告: 未找到 prompt.txt 文件，使用默认提示词")
            self.system_prompt = "你是一个有用的AI助手，可以使用各种工具来帮助用户完成任务。"
        
        # 创建提示模板
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建聊天历史
        self.chat_history = InMemoryChatHistory()
        
        # 回调处理器
        self.callback = MCPCallback()
        
        # Agent和Executor（稍后初始化）
        self.agent = None
        self.agent_executor = None
    
    async def connect_to_server(self, command: str, server_id: str, args: List[str]):
        """连接到MCP服务器"""
        if server_id in self.sessions:
            raise ValueError(f"服务端 {server_id} 已经连接")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        # 启动MCP服务器
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )

        await session.initialize()
        self.sessions[server_id] = {
            "session": session, 
            "stdio": stdio, 
            "write": write
        }
        print(f"已连接到 MCP 服务器: {server_id}")

        # 获取工具并创建LangChain工具
        response = await session.list_tools()
        for tool in response.tools:
            self.tools_map[tool.name] = server_id
            
            # 创建LangChain工具
            mcp_tool = MCPTool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
                client_session=session,
                server_id=server_id
            )
            self.mcp_tools.append(mcp_tool)
    
    def initialize_agent(self):
        """初始化LangChain Agent"""
        if not self.mcp_tools:
            print("警告: 没有可用的MCP工具，创建无工具的Agent")
        
        # 创建Agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.mcp_tools,
            prompt=self.prompt_template
        )
        
        # 创建Agent执行器（简化版，不使用已弃用的memory参数）
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.mcp_tools,
            callbacks=[self.callback],
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
    
    async def list_tools(self):
        """列出所有可用工具"""
        if not self.mcp_tools:
            print("没有可用的工具")
            return
        
        print("可用工具列表:")
        for tool in self.mcp_tools:
            print(f"- {tool.name}: {tool.description}")
    
    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        try:
            # 手动管理聊天历史
            self.chat_history.add_message(HumanMessage(content=query))
            
            # 使用Agent执行器处理查询
            response = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": self.chat_history.messages
            })
            
            # 添加AI响应到历史
            self.chat_history.add_message(AIMessage(content=response["output"]))
            
            return response["output"]
            
        except Exception as e:
            print(f"处理查询时发生错误: {e}")
            return f"抱歉，处理您的查询时发生错误: {str(e)}"
    
    def extract_and_save_code(self, content: str) -> bool:
        """提取并保存代码块"""
        try:
            with open("output.py", "w", encoding="utf-8") as f:
                inside_code_block = False
                for line in content.split("\n"):
                    if line.startswith("```python"):
                        inside_code_block = True
                        continue
                    if line.startswith("```") and inside_code_block:
                        inside_code_block = False
                        continue
                    if inside_code_block:
                        f.write(line + "\n")
            
            print("代码提取并保存成功!")
            return True
            
        except Exception as e:
            print(f"提取代码失败: {e}")
            return False
    
    def run_extracted_code(self):
        """运行提取的代码"""
        try:
            interpreter_path = "E:/tsinghua/aconda/envs/s/python.exe"
            script_path = "output.py"
            return_code = run_python_file(interpreter_path, script_path)
            
            if return_code == 0:
                print("代码执行成功!")
            return return_code == 0
            
        except Exception as e:
            print(f"运行代码失败: {e}")
            return False
    
    async def chat_loop(self):
        """交互式聊天循环"""
        print("\nLangChain MCP 客户端已启动!")
        print("请输入您的问题，输入'quit'退出，输入'new'重新开始。")
        
        while True:
            try:
                query = input("\n问题: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'new':
                    self.chat_history.clear()
                    print("对话历史已清除!")
                    continue
                
                # 处理查询
                response = await self.process_query(query)
                print(f"\n回答: {response}")
                
                # 提取并执行代码
                if self.extract_and_save_code(response):
                    run_code = input("\n是否运行提取的代码? (y/n): ")
                    if run_code.lower() == 'y':
                        self.run_extracted_code()
                
            except KeyboardInterrupt:
                print("\n用户中断，退出...")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    """主函数"""
    client = LangChainMCPClient()
    
    try:
        # 加载配置并连接服务器
        try:
            with open("config.json", "r", encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            print("错误: 未找到 config.json 文件")
            print("请创建配置文件，示例格式:")
            print("""
{
  "mcpServers": {
    "example_server": {
      "command": "python",
      "args": ["path/to/server.py"]
    }
  }
}
            """)
            return
        except json.JSONDecodeError as e:
            print(f"错误: config.json 文件格式错误: {e}")
            print("请检查JSON格式，确保:")
            print("1. 所有字符串都用双引号包围")
            print("2. 没有多余的逗号")
            print("3. 括号正确匹配")
            return
            
        for server_name, server_info in config["mcpServers"].items():
            print(f"正在连接到服务器: {server_name}")
            await client.connect_to_server(
                server_info["command"], 
                server_name, 
                server_info["args"]
            )
        
        # 初始化Agent
        client.initialize_agent()
        
        # 列出工具
        await client.list_tools()
        
        # 启动聊天循环
        await client.chat_loop()
        
    finally:
        try:
            await client.cleanup()
        except Exception as e:
            print(f"清理时发生错误: {e}")
        
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
    
    # XODR文件运行选项
    run_xodr = input("是否运行xodr文件? (y/n): ")
    if run_xodr.lower() == 'y':
        subprocess.run([
            r"E:/tsinghua/esmini/bin/odrviewer.exe", 
            "--odr", 
            r"E:/tsinghua/code/mcp-client/xodr/output0.xodr"
        ], check=True)
        print("已运行 xodr 文件。")
    else:
        print("已退出，未运行 xodr 文件。")