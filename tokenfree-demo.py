import requests
import json
import os
from dotenv import load_dotenv
import base64
from typing import Dict, Any

# 加载 .env 文件
load_dotenv()
TOKENFREE_TOKEN = os.getenv("TOKENFREE_TOKEN")
API_URL = "https://api.tokenfree.ai/v1/chat/completions"


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_payload(image_base64: str) -> Dict[str, Any]:
    return {
        "model": "llava-onevision-qwen2-72b-ov",
        "messages": [
            {"role": "system", "content": "你是一个中文智者，请用中文回答问题"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "用中文描述一下这张图片"},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            },
        ],
        "max_tokens": 512,
        "top_p": 1,
        "temperature": 0.1,
        "stream": False,
    }


def send_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKENFREE_TOKEN}",
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def main():
    image_path = "images/cat (1).jpg"
    image_base64 = image_to_base64(image_path)
    payload = create_payload(image_base64)

    try:
        result = send_request(payload)
        content = result["choices"][0]["message"]["content"]
        print(content)
    except requests.RequestException as e:
        print(f"请求失败: {e}")
    except KeyError as e:
        print(f"解析响应失败: {e}")


if __name__ == "__main__":
    main()
