import requests
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {os.getenv('API_TOKEN')}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not in JSON format.")
        return None


data = query({"inputs": "Can you please let us know more details about your"})

print(data)
