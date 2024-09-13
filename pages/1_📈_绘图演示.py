import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="绘图演示", page_icon="📈")

st.markdown("# 绘图演示")
st.sidebar.header("绘图演示")
st.write(
    """这个演示展示了使用 Streamlit 进行绘图和动画的组合。
我们将循环生成一堆随机数，持续大约 5 秒钟。请欣赏！"""
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% 完成" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit 小部件自动从上到下运行脚本。由于此按钮未连接到其他逻辑，
# 它只是导致简单的重新运行。
st.button("重新运行")
