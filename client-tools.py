import subprocess
import sys
import argparse

def run_python_file(interpreter_path, script_path, script_args=None):
    """
    在指定的Python环境中运行Python文件
    
    参数:
    interpreter_path (str): Python解释器的路径
    script_path (str): 要运行的Python脚本的路径
    script_args (list, 可选): 传递给Python脚本的参数列表
    
    返回:
    int: 子进程的返回码
    """
    # 构建命令
    command = [interpreter_path, script_path]
    
    # 添加脚本参数（如果有）
    if script_args:
        command.extend(script_args)
    
    try:
        # 执行命令
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印标准输出
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
        
        return result.returncode
    
    except subprocess.CalledProcessError as e:
        # 打印错误信息
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
    

import asyncio # 用于导入异步IO库，用于支持异步编程
import json # 用于导入JSON库，用于处理JSON数据
import sys # 用于处理命令行参数
from typing import Optional # 用于类型提示功能
from contextlib import AsyncExitStack # 异步资源管理器，用于管理多个异步资源

# MCP 客户端相关导入
from mcp import ClientSession, StdioServerParameters # 导入 MCP 客户端会话和标准输入输出服务器参数
from mcp.client.stdio import stdio_client # 导入标准输入输出客户端通信模块

from openai import OpenAI # Openai SDK

# 环境变量加载相关
from dotenv import load_dotenv # 导入环境变量加载工具
import os # 用于获取环境变量值

# 从promt.txt文件中读取系统提示词
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read().strip()  # 读取文件内容并去除首尾空白字符

load_dotenv()  # 加载 .env 文件中的环境变量

# 定义 MCP 客户端类
class DeepSeekMCPClient:
    """
    使用 DeepSeek V3 API 的 MCP 客户端类
    处理 MCP 服务器连接和 DeepSeek V3 API 的交互
    """
    def __init__(self):
        """ 
        初始化MCP客户端的各项属性
        """
        # MCP 客户端会话，初始值为 None
        self.sessions = {}
        # 创建异步资源管理器，用于管理多个异步资源
        self.exit_stack = AsyncExitStack()
        # 初始化 DeepSeek API 客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("API_KEY"), # 从环境变量中获取 API 密钥
            base_url=os.getenv("BASE_URL") # 从环境变量中获取 API 基础 URL
        )
        # 从环境变量获取模型名称
        self.model = os.getenv("MODEL")
        self.tools_map = {} # 工具映射字典，用于存储工具名称和对应的服务器标识符
    
    async def connect_to_server(self, command, server_id: str, args: str):
        """
        连接到 MCP 服务器
        :param server_id: 服务端标识符
        :param server_script_path: 服务端脚本路径
        """
        if server_id in self.sessions:
            raise ValueError(f"服务端 {server_id} 已经连接")

        server_params = StdioServerParameters(command=command,
                                              args=args,
                                              env=None)

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await session.initialize()
        self.sessions[server_id] = {"session": session, "stdio": stdio, "write": write}
        print(f"已连接到 MCP 服务器: {server_id}")

        # 更新工具映射
        response = await session.list_tools()
        for tool in response.tools:
            self.tools_map[tool.name] = server_id
    
    async def list_tools(self):
        if not self.sessions:
            print("没有已连接的服务端")
            return

        print("已连接的服务端工具列表:")
        for tool_name, server_id in self.tools_map.items():
            print(f"工具: {tool_name}, 来源服务端: {server_id}")

    async def process_query(self, query: str) -> str:
        """
        处理用户查询，根据查询参数使用DeepSeek V3和MCP工具
        """
        # 创建消息列表，用于存储用户的问题和模型的回答
        messages = [
            {
                "role": "system", # 系统角色，用于设定AI的行为准则
                "content": prompt # 系统提示词，指导模型如何回答问题
            }
        ]
        messages.extend(query) # 将用户的问题添加到消息列表中
        # 构建统一的工具列表
        available_tools = []
        for tool_name, server_id in self.tools_map.items():
            session = self.sessions[server_id]["session"]
            response = await session.list_tools()
            for tool in response.tools:
                if tool.name == tool_name:
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema
                        }
                    })
        print('整合的服务端工具列表:', available_tools)

        # 循环调用
        while True:
        # 调用 DeepSeek API，发送用户查询和可用工具信息，告诉 DeepSeek API 根据用户提问你可以使用哪些工具，最终返回可调用的工具
            response = self.llm_client.chat.completions.create(
                model=self.model, # 指定的模型名称
                messages=messages, # 消息历史（系统提示和用户问题）
                tools=available_tools if available_tools else None, # 可用的工具列表
                temperature=0.5, # 温度参数，控制响应的随机性(0.5是中等随机性)
                max_tokens=4096 # 最大生成令牌数，限制响应长度
                )
            # 打印模型响应，便于调试
            print(f"DeepSeek API 响应: {response}\n--------------------------------\n")
            # 获取模型的回复，包含 role(消息发送者) 和 content(消息内容) 以及 tool_calls(工具调用请求)
            reply = response.choices[0].message # 获取模型的回答
            # 打印模型的回答
            print(f"DeepSeek 初始回复: {reply}\n--------------------------------\n")
            
            # 初始化最终文本结果列表
            final_text = []

            # 将模型回复添加到历史消息中，用于维护完整的对话历史
            # 确保模型记得自己之前决定使用什么工具，即使模型没有请求调用工具，也要保持对话连贯性。
            messages.append(reply)

            # 检查模型响应中是否包含工具调用请求，如果用户的问题涉及到使用工具，那就会包含 tool_calls 字段,否则就没有
            if hasattr(reply, "tool_calls") and reply.tool_calls:
                # 遍历所有工具调用请求
                for tool_call in reply.tool_calls:
                    # 获取工具名称
                    tool_name = tool_call.function.name
                    # 获取工具参数
                    try:
                        # 尝试将工具的参数从 JSON 字符串解析为 Python 字典
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}
                    server_id = self.tools_map.get(tool_name)
                    if not server_id:
                        raise ValueError(f"未找到{tool_name}对应的服务端")
                    
                    # 打印工具调用信息，便于调试
                    print(f"准备调用工具: {tool_name} 参数: {tool_args}\n--------------------------------\n")

                    # 异步调用 MCP 服务上的工具，传入工具名称和函数参数，返回工具函数执行结果
                    # 通过工具名称获取对应的服务端ID
                    session = self.sessions[server_id]["session"]
                    result = await session.call_tool(tool_name, tool_args)
                    # 打印工具执行结果，便于调试
                    print(f"工具 {tool_name} 执行结果: {result}\n--------------------------------\n")

                    # 将工具调用信息添加到最终输出文本中，便于用户了解执行过程
                    final_text.append(f"调用工具: {tool_name}, 参数: {tool_args}\n")

                    # 确保工具结果是字符串格式
                    tool_result_content = result.content
                    if isinstance(tool_result_content, list):
                        # 如果工具结果是列表，则将列表中的每个元素转换为字符串并添加到最终文本中
                        text_content = ""
                        for item in tool_result_content:
                            if hasattr(item, 'text'):
                                text_content += item.text
                        tool_result_content = text_content
                    elif not isinstance(tool_result_content, str):
                        # 如果不是字符串，则转换为字符串
                        tool_result_content = str(tool_result_content)
                    # 打印工具返回结果
                    print(f"工具返回结果(格式化后): {tool_result_content}\n--------------------------------\n")

                    # 将工具调用结果添加到历史消息中，保证与模型会话的连贯性
                    tool_message = {
                        "role": "tool", # 工具角色，表示这是工具返回的结果
                        "tool_call_id": tool_call.id, # 工具调用ID
                        "content": tool_result_content, # 工具返回的结果
                        
                    }
                    # 打印消息内容
                    print(f"添加到历史消息中的工具消息: {tool_message}\n--------------------------------\n")
                    # 添加到历史消息中
                    messages.append(tool_message)

                    # 尝试解析工具返回的JSON结果，检查是否包含MCP模板结构
                    try:
                        # 将工具返回结果 JSON格式 转换为 Python 字典
                        tool_result_json = json.loads(tool_result_content)
                        # 检查是否包含 MCP 模板结构(具有 prompt_template 和 template_args 字段)
                        if(isinstance(tool_result_json, dict) and "prompt_template" in tool_result_json and "template_args" in tool_result_json):
                            raw_data = tool_result_json["raw_data"] # 原始数据
                            prompt_template = tool_result_json["prompt_template"] # 模板函数名称
                            template_args = tool_result_json["template_args"] # 模板参数

                            # 将模板参数转换为字符串类型(MCP规范要求)
                            string_args = {k:str(v) for k,v in template_args.items()}
                            # 打印模板参数
                            print(f"模板名称: {prompt_template}, 模板参数: {string_args}\n--------------------------------\n")

                            # 调用 MCP 服务上的工具，传入工具名称和函数参数，返回工具函数执行结果
                            template_response = await self.sessions.get_prompt(prompt_template, string_args)
                            # 打印工具执行结果，便于调试
                            print(f"模板响应: {template_response}\n--------------------------------\n")

                            if hasattr(template_response, "messages") and template_response.messages:
                                # 打印模板响应
                                print(f"模板具体的信息: {template_response.messages}\n--------------------------------\n")
                                for msg in template_response.messages:
                                    # 提取消息内容
                                    content = msg.content.text if hasattr(msg.content, "text") else msg.content
                                    # 构建历史信息
                                    template_message = {
                                        "role": msg.role, # 保持原始角色
                                        "content": content # 消息内容
                                    }
                                    print(f"模板消息历史: {template_message}\n--------------------------------\n")
                                    # 添加到历史消息中
                                    messages.append(template_message)
                            else:
                                print("警告：模板响应中没有包含消息内容。")
                    except json.JSONDecodeError:
                        pass
                    # 再次调用 DeepSeek API，让模型根据工具结果生成最终的回答
                    try:
                        print("正在请求 DeepSeek API 生成回答...")
                        # 发送包含工具调用和结果的完整消息历史
                        final_response = self.llm_client.chat.completions.create(
                            model=self.model, # 指定的模型名称
                            messages=messages, # 消息历史（系统提示和用户问题）
                            tools=available_tools if available_tools else None, # 可用的工具列表
                            temperature=0.5, # 温度参数，控制响应的随机性(0.5是中等随机性)
                            max_tokens=4096 # 最大生成令牌数，限制响应长度
                        )
                        # 添加 DeepSeek 对工具结果的解释然后到最终输出
                        final_content = "DeepSeek回答：" + final_response.choices[0].message.content
                        if final_content:
                            # 如果模型生成了对工具结果的解释，就将其添加到最终输出数组中
                            final_text.append(final_content)
                            print(messages)
                        else:
                            print("警告：DeepSeek API 没有生成任何内容。")
                            # 如果没用内容，直接显示工具结果
                            final_text.append(f"工具调用结果：\n{tool_result_content}")
                        print(f"中间回答：{messages}")
                    except Exception as e:
                        print(f"生成最终回复时出错: {e}")
                continue

            else:
                # 如果模型没有请求调用工具，那么就直接返回模型的内容
                if reply.content:
                    # 将模型的直接回复添加到最终输出数组
                    final_text.append(f"{reply.content}")
                else:
                    # 如果模型没有生成内容，则添加提示信息
                    final_text.append("模型没有生成有效回复。")
        
            # 返回最终的回答
                print(f"============最终回答=============")
                return final_text,'\n'.join(final_text)
    
    async def chat_loop(self):
        """
        运行交互式聊天循环，处理用户输入并显示回复
        
        这个函数就是一个简单的聊天界面，不断接收用户输入，
        处理问题，并显示回答，直到用户输入'quit'退出。
        """
        print("\nDeepSeek MCP 客户端已经启动!")
        print("请输入你的问题，输入'quit'退出，输入'new'重新开始。")
        # 循环处理用户输入
        history = []
        while True:
            try:
                # 获取用户输入
                query = input("\n问题: ").strip()
                # 检查是否要退出
                if query.lower() == 'new':
                    # 如果用户输入 'new'，则重新开始
                    print("重新开始...")
                    history = []
                    print("\n=================重置历史消息成功!=================\n")
                    continue
                if query.lower() == 'quit':
                    break
                # 处理用户输入，传入到查询函数中
                history.append({"role": "user", "content": query})
                print(history)
                response,re = await self.process_query(history)
                print("\n"+re)
                history.extend([{"role": "assistant", "content": response[0].replace('</think>\n\n', '')}])
                # 提取并保存文档中 ```python 和 ``` 之间的内容
                try:
                    with open("output.py", "w", encoding="utf-8") as f:
                        print(f"输出文件内容: {re}")
                        inside_code_block = False  # 标记是否在代码块中
                        for line in re.split("\n"):  # 将内容按行分割
                            if line.startswith("```python"):
                                inside_code_block = True  # 开始代码块
                                continue
                            if line.startswith("```") and inside_code_block:
                                inside_code_block = False  # 结束代码块
                                continue
                            if inside_code_block:
                                f.write(line + "\n")  # 写入代码块内容
                    print("\n============输出文件内容成功!================\n")
                    # 运行输出的 Python 文件
                    try:
                        interpreter_path = "E:/tsinghua/aconda/envs/s/python.exe"  # 获取当前 Python 解释器的路径
                        script_path = "output.py"  # 输出的 Python 文件路径
                        return_code = run_python_file(interpreter_path, script_path)
                        if return_code == 0:
                            print("输出的 Python 文件执行成功!")
                    except Exception as e:
                        print(f"运行输出的 Python 文件失败: {str(e)}")
                except Exception as e:
                    print(f"\n输出文件内容失败: {str(e)}")
                print("\n============历史消息================\n")
                print(history)

            except Exception as e:
                print(f"\n错误: {str(e)}")

    async def cleanup(self):
        """
        清理资源，关闭所有打开的连接和上下文。
        """
        # 关闭所有打开的连接和上下文,释放资源
        await self.exit_stack.aclose()

async def main():
    """
    主函数，处理命令行参数并启动客户端
    """
    # 检查命令行参数
    # 创建客户端实例
    client = DeepSeekMCPClient()
    try:
        # 连接到MCP服务器
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            for server_name, server_info in config["mcpServers"].items():
                print(f"正在连接到服务器: {server_name}")
                # 连接到指定的 MCP 服务器脚本
                await client.connect_to_server(server_info["command"], server_name,server_info["args"])
        # 启动聊天循环
        await client.list_tools()
        await client.chat_loop()
    finally:
        # 清理资源,确保在任何情况下都清理资源
        try:
            await client.cleanup()
        except Exception as e:
            print(f"清理时发生错误: {e}")
        # 确保事件循环完全关闭
        await asyncio.sleep(1)

# 程序入口点
if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())

    input = input("是否运行xodr文件? (y/n): ")
    if input.lower() == 'y':

        # 运行 cd 命令和 esmini 命令
        # 注意：在 Windows 上，cd 命令不能直接在 subprocess 中使用，需要使用 shell=True 或者直接在命令行中执行
        subprocess.run(
            [r"E:/tsinghua/esmini/bin/odrviewer.exe", "--odr", r"E:/tsinghua/code/mcp-client/xodr/output0.xodr"],
            check=True
        )
        print("已运行 xodr 文件。")
    else:
        print("已退出，未运行 xodr 文件。")
# 使用说明
# 激活虚拟环境（如果尚未激活）
# .venv\Scripts\activate

# 运行 MCP 客户端，连接到天气查询 MCP 服务器（示例）
# uv run client-tools.py 
