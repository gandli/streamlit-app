import requests
import os
import json
from dotenv import load_dotenv
import base64
from typing import Dict, Any, Generator, List

# 加载 .env 文件
load_dotenv()
TOKENFREE_TOKEN = os.getenv("TOKENFREE_TOKEN")
API_URL = "https://api.tokenfree.ai/v1/chat/completions"

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_payload(image_base64: str, text: str) -> Dict[str, Any]:
    return {
        "model": "llava-onevision-qwen2-72b-ov",
        "messages": [
            {"role": "system", "content": "你是一个中文智者，请用中文回答问题"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            },
        ],
        "max_tokens": 512,
        "top_p": 1,
        "temperature": 0.1,
        "stream": True,
    }

def stream_request(payload: Dict[str, Any]) -> Generator[str, None, None]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKENFREE_TOKEN}",
    }
    with requests.post(API_URL, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                yield line.decode("utf-8")[6:]

def describe_image(image_path: str, text: str) -> List[str]:
    image_base64 = image_to_base64(image_path)
    payload = create_payload(image_base64, text)
    descriptions = []

    try:
        for chunk in stream_request(payload):
            try:
                chunk_data = json.loads(chunk)
                if "choices" in chunk_data:
                    for choice in chunk_data["choices"]:
                        if "delta" in choice and "content" in choice["delta"]:
                            content = choice["delta"]["content"]
                            if content is not None:
                                descriptions.append(content)
            except json.JSONDecodeError:
                pass
    except requests.RequestException as e:
        print(f"请求失败: {e}")

    return descriptions

# Example usage:
if __name__ == "__main__":
    image_path = "data/images/cat/cat (1).jpg"
    text = "用中文详细地描述一下这张图片，最后描述这张图片的关键词，不要有任何前缀"
    descriptions = describe_image(image_path, text)
    print("".join(descriptions))
