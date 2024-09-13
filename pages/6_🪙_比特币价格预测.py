import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(page_title="比特币价格预测", page_icon="📈")

# 加载数据集
@st.cache_data
def load_data():
    data = pd.read_csv("data/challenge-2-bitcoin.csv", parse_dates=["Date"])
    return data

data = load_data()

# 选择特征和目标值
selected_features = ["btc_total_bitcoins", "btc_transaction_fees"]
target = "btc_market_price"

# 数据预览
st.write("# 比特币价格预测 📈")
st.write("### 数据预览")
st.write(data.head())

# 数据集说明
st.write("# 比特币数据集说明")

# 特征说明
features = {
    "Date": "日期",
    "btc_market_price": "比特币市场价格（美元）",
    "btc_total_bitcoins": "比特币总量",
    "btc_market_cap": "比特币市场总值（美元）",
    "btc_trade_volume": "比特币交易量（美元）",
    "btc_blocks_size": "比特币区块大小（字节）",
    "btc_avg_block_size": "比特币平均区块大小（MB）",
    "btc_n_orphaned_blocks": "比特币孤块数",
    "btc_n_transactions_per_block": "每个区块的比特币交易数",
    "btc_median_confirmation_time": "比特币确认时间中位数（分钟）",
    "btc_hash_rate": "比特币哈希率（TH/s）",
    "btc_difficulty": "比特币挖矿难度",
    "btc_miners_revenue": "比特币矿工收入（美元）",
    "btc_transaction_fees": "比特币交易费用（美元）",
    "btc_cost_per_transaction_percent": "每笔交易的成本占比（%）",
    "btc_cost_per_transaction": "每笔交易的成本（美元）",
    "btc_n_unique_addresses": "比特币独立地址数量",
    "btc_n_transactions": "比特币交易数量",
    "btc_n_transactions_total": "比特币总交易数量",
    "btc_n_transactions_excluding_popular": "排除热门地址的比特币交易数量",
    "btc_n_transactions_excluding_chains_longer_than_100": "排除长度超过 100 的链的比特币交易数量",
    "btc_output_volume": "比特币输出量",
    "btc_estimated_transaction_volume": "估计的比特币交易量",
    "btc_estimated_transaction_volume_usd": "估计的比特币交易量（美元）",
}

# 将特征和说明转为一个 DataFrame
features_df = pd.DataFrame(list(features.items()), columns=["特征", "说明"])

# 显示特征说明
st.write("### 数据集特征说明")
st.dataframe(features_df, use_container_width=True)

# 数据预处理
X = data[selected_features]
y = data[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 构建和评估多项式回归模型
degrees = [1, 2, 3, 4, 5]  # 不同的多项式次数
mse_train = []
mse_test = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

# 绘制 MSE 与多项式次数的关系图
plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_train, marker="o", label="训练集 MSE")
plt.plot(degrees, mse_test, marker="s", label="测试集 MSE")
plt.xlabel("多项式次数")
plt.ylabel("均方误差 (MSE)")
plt.title("不同多项式次数下的均方误差 (MSE)")
plt.legend()
plt.grid(True)

# 显示绘图结果
st.write("### 不同多项式次数下的均方误差 (MSE)")
st.pyplot(plt)

# 选择最佳多项式次数
best_degree = degrees[np.argmin(mse_test)]
st.write(f"最佳多项式次数: {best_degree}")

# 使用最佳多项式次数训练模型并预测
poly = PolynomialFeatures(best_degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_test_pred = model.predict(X_test_poly)

# 显示预测结果
st.write("### 预测结果")
results = pd.DataFrame({"真实值": y_test, "预测值": y_test_pred})
st.write(results.head())

# 绘制预测结果图表
st.write("### 预测结果图表")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
ax.set_xlabel("真实值")
ax.set_ylabel("预测值")
ax.set_title("预测值与真实值对比")
st.pyplot(fig)
