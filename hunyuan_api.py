import json
import os
from dotenv import load_dotenv
from typing import Dict, Generator
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

# 加载 .env 文件
load_dotenv()


def hunyuan_generator(model_name: str, messages: Dict) -> Generator:
    """
    调用腾讯云混元大模型 API，并支持流式输出。

    :param model_name: 模型名称，例如 "hunyuan-standard"
    :param messages: 聊天历史记录，格式为 [{"role": "user/assistant/system/tool", "content": "message content"}]
    :yield: 逐步生成的模型响应内容
    """
    try:
        # 实例化一个认证对象
        cred = credential.Credential(
            os.getenv("TENCENTCLOUD_SECRET_ID"),  # 从 .env 文件读取 secretId
            os.getenv("TENCENTCLOUD_SECRET_KEY"),  # 从 .env 文件读取 secretKey
        )

        cpf = ClientProfile()
        cpf.httpProfile.pre_conn_pool_size = 3
        client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou", cpf)

        req = models.ChatCompletionsRequest()
        req.Model = model_name
        req.Stream = True

        # 构建消息列表，确保角色字段为有效值
        req.Messages = []
        valid_roles = ["system", "user", "assistant", "tool"]
        for msg in messages:
            if msg["role"] not in valid_roles:
                raise ValueError(
                    f"Invalid role: {msg['role']}. Must be one of {valid_roles}."
                )

            message = models.Message()
            message.Role = msg["role"]
            message.Content = msg["content"]
            req.Messages.append(message)

        # 流式输出
        resp = client.ChatCompletions(req)
        for event in resp:
            data = json.loads(event["data"])
            for choice in data["Choices"]:
                yield choice["Delta"]["Content"]

    except TencentCloudSDKException as err:
        yield f"Error: {str(err)}"


def get_available_models() -> list:
    """
    获取可用的混元模型列表。

    :return: 可用模型的名称列表
    """
    return [
        "hunyuan-lite",
        "hunyuan-standard",
        "hunyuan-standard-256K",
        "hunyuan-pro",
        "hunyuan-code",
        "hunyuan-role",
        "hunyuan-functioncall",
        "hunyuan-vision",
    ]  # 可以根据实际可用模型更新此列表


# 示例调用
if __name__ == "__main__":
    # 示例聊天记录
    conversation = [
        {"role": "user", "content": "请介绍下自己"},
        {
            "role": "assistant",
            "content": "我是腾讯混元大模型，擅长提供问答服务和常识推理等服务。如果您需要帮助或有任何问题，请随时向我提问。",
        },
        {"role": "user", "content": "请给我讲个搞笑笑话"},
    ]

    # 流式输出示例
    for content in hunyuan_generator("hunyuan-lite", conversation):
        print(content, end="")  # 使用 end="" 使内容逐步输出，而不是换行

    print("\n")  # 换行以分隔输出
