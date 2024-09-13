import os
from sklearn.datasets import fetch_lfw_people

# 设置代理（根据需要调整）
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


def load_data():
    try:
        # 尝试加载数据集
        faces = fetch_lfw_people(min_faces_per_person=60)
        print("目标类别名称:", faces.target_names)
        print("数据集图像形状:", faces.images.shape)
        return faces
    except ValueError as e:
        print(f"数据加载出错: {e}")
        return None


# 加载数据集
faces = load_data()
