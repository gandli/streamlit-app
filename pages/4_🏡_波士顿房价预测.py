import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 页面配置
st.set_page_config(
    page_title="波士顿房价预测",
    page_icon="🏡",
)



# 加载数据集
@st.cache_data
def load_data():
    data = pd.read_csv("data/course-5-boston.csv")
    return data


data = load_data()

# 显示数据集
st.write("# 波士顿房价预测 🏡")
st.write("### 数据预览")
st.write(data.head())

# 数据集说明
st.write("# 波士顿房价数据集说明")
features = {
    "CRIM": "镇人均犯罪率",
    "ZN": "占地面积超过 25,000 平方英尺的住宅用地比例",
    "INDUS": "每个镇的非零售业务用地比例",
    "CHAS": "查尔斯河虚拟变量（如果区域边界是河流则为1，否则为0）",
    "NOX": "氮氧化物浓度（每 1000 万分之一）",
    "RM": "每个住宅的平均房间数",
    "AGE": "1940年之前建造的自住房屋比例",
    "DIS": "到波士顿五个就业中心的加权距离",
    "RAD": "辐射公路的可达性指数",
    "TAX": "每 $10,000 美元的全额物业税率",
    "PTRATIO": "每个镇的师生比",
    "BLACK": "1000(Bk - 0.63)^2，其中 Bk 是每个镇的黑人比例",
    "LSTAT": "低收入人口的百分比",
    "MEDV": "自住房屋的中位数价值（以 $1000 为单位）",
}

# 将特征和说明转为一个 DataFrame
features_df = pd.DataFrame(list(features.items()), columns=['特征', '说明'])

# 显示特征说明
st.write("### 数据集特征说明")
st.dataframe(features_df, use_container_width=True)

# 显示数据统计信息
st.write("### 数据统计信息")
st.write(data.describe())

# 数据预处理
X = data.drop("medv", axis=1)
y = data["medv"]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = LinearRegression()  # 建立模型
model.fit(X_train, y_train)  # 训练模型

# 预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
st.write("### 预测结果")
st.write(f"均方误差: {mse:.2f}")

# 显示预测和真实值对比
results = pd.DataFrame({"真实值": y_test, "预测值": y_pred})
st.write(results.head())

# 绘制预测结果图表
st.write("### 预测结果图表")
st.line_chart(results)
