import ollama
from typing import Dict, Generator


def ollama_generator(model_name: str, messages: Dict) -> Generator:
    """
    调用 Ollama 模型，并支持流式输出。

    :param model_name: 使用的模型名称
    :param messages: 聊天历史记录，格式为 [{"role": "user/assistant", "content": "message content"}]
    :yield: 逐步生成的模型响应内容
    """
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]


def get_available_models() -> list:
    """
    获取可用的 Ollama 模型列表。

    :return: 可用模型的名称列表
    """
    return [model["name"] for model in ollama.list()["models"]]
