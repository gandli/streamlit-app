import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
import numpy as np
import tensorflow as tf


# 加载模型
@st.cache_resource
def load_letter_model():
    return tf.keras.models.load_model(
        "models/CNN_MNIST_20240911_epochs_50_batch_size_32.h5", compile=False
    )


model = load_letter_model()

st.title("手写识别 - 字母 🔠")

# 创建两列布局
col1, col2 = st.columns(2)

# 左侧画布组件
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=10,
        stroke_color="rgba(0, 0, 0, 1)",
        background_color="#FFFFFF",
        update_streamlit=True,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas_letter",
    )

# 右侧按钮和识别结果
with col2:
    if st.button("识别", key="recognize_letter"):
        if canvas_result.image_data is not None:
            # 处理绘制的图像
            gray_image = rgb2gray(rgba2rgb(canvas_result.image_data))
            resized_image = resize(gray_image, (28, 28), anti_aliasing=True)
            processed_image = np.abs(1 - resized_image).reshape(1, 28, 28)

            # 显示预测中的提示
            prediction_status = st.empty()
            prediction_status.write("预测中...")

            # 模型预测
            predictions = chr(65 + np.argmax(model.predict(processed_image)))

            # 更新预测结果
            prediction_status.write(f"## 识别结果: {predictions[0]}")
            
            st.image(resized_image, caption="调整大小后的图像", width=150)
        else:
            st.warning("请先在画布上绘制字母。")

# 添加使用说明
st.markdown("""
## 使用说明
1. 在左侧画布上用鼠标绘制一个大写字母（A-Z）。
2. 点击"识别"按钮进行预测。
3. 系统将显示识别结果和调整大小后的图像。
""")
