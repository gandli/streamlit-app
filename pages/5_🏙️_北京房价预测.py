import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 页面配置
st.set_page_config(
    page_title="北京房价预测",
    page_icon="🏙️",
)


# 加载数据集
@st.cache_data
def load_data():
    data = pd.read_csv("data/challenge-1-beijing.csv")
    return data


data = load_data()

# 显示数据集
st.write("# 北京房价预测 🏙️")
st.write("### 数据预览")
st.write(data.head())

# 数据集说明
st.write("# 北京房价数据集说明")
features = {
    "公交": "距离最近的公交站点数量",
    "写字楼": "距离最近的写字楼数量",
    "医院": "距离最近的医院数量",
    "商场": "距离最近的商场数量",
    "地铁": "距离最近的地铁站点数量",
    "学校": "距离最近的学校数量",
    "小区名字": "小区的名称",
    "建造时间": "房屋建造的年份",
    "房型": "房屋的类型（如 2室1厅、3室2厅等）",
    "楼层": "房屋所在的楼层",
    "每平米价格": "房屋每平方米的价格（单位：元）",
    "面积": "房屋的总面积（单位：平方米）",
}

# 将特征和说明转为一个 DataFrame
features_df = pd.DataFrame(list(features.items()), columns=["特征", "说明"])

# 显示特征说明
st.write("### 数据集特征说明")
st.dataframe(features_df, use_container_width=True)

# 显示数据统计信息
st.write("### 数据统计信息")
st.write(data.describe())

# 数据预处理
# 假设 '小区名字', '房型', '楼层' 为类别特征，需要进行编码
data = pd.get_dummies(data, columns=["小区名字", "房型", "楼层"], drop_first=True)

# 将'建造时间'转换为数值型数据
data["建造时间"] = pd.to_numeric(data["建造时间"], errors="coerce")

# 移除缺失值
data = data.dropna()

# 增加更多特征
data["建造时间平方"] = data["建造时间"] ** 2
data["面积平方"] = data["面积"] ** 2
data["公交_写字楼"] = data["公交"] * data["写字楼"]
data["医院_地铁"] = data["医院"] * data["地铁"]
data["商场_学校"] = data["商场"] * data["学校"]

# 特征和目标值
X = data.drop("每平米价格", axis=1)
y = data["每平米价格"]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 显示预测结果
st.write("### 预测结果")
st.write(f"平均绝对误差 (MAE): {mae:.2f}")
st.caption(
    f"MAE表示预测值与真实值之间的平均绝对差异。对于这个数据集，MAE为{mae:.2f}。这意味着，模型预测的房价每平米价格平均与真实值相差约{mae:.2f}元。"
)

st.write(f"均方误差 (MSE): {mse:.2f}")
st.caption(
    f"MSE表示预测值与真实值之间的平方差异的平均值，MSE为{mse:.2f}。由于平方误差放大了大偏差的影响，因此MSE较大。MSE值越低，模型的预测结果越好。"
)

st.write(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
st.caption(
    f"MAPE表示预测值与真实值之间的平均绝对百分比差异，对于这个数据集，MAPE为{mape:.2f}%。这意味着，模型的预测值平均比真实值偏差了{mape:.2f}%。在房价预测中，{mape:.2f}% 的 MAPE 表明模型有一定的预测能力，但仍有提升的空间。"
)

# 显示预测和真实值对比
results = pd.DataFrame({"真实值": y_test, "预测值": y_pred})
st.write(results.head())

# 绘制预测结果图表
st.write("### 预测结果图表")
st.line_chart(results)
