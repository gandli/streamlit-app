import streamlit as st
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.transform import resize
import joblib


def load_models():
    """加载保存的模型"""
    svm_clf = joblib.load("data/svm_clf_model.joblib")
    scaler = joblib.load("data/scaler_model.joblib")
    return svm_clf, scaler


def preprocess_image(image, target_shape):
    """预处理上传的图片"""
    image_resized = resize(image, target_shape, mode="reflect", anti_aliasing=True)
    image_gray = np.dot(
        image_resized[..., :3], [0.2989, 0.5870, 0.1140]
    )  # Convert to grayscale
    return image_gray.flatten()


st.title("LFW 人脸分类")

faces = fetch_lfw_people(
    data_home="data/lfw_funneled", min_faces_per_person=60, resize=0.4
)

tab1, tab2, tab3 = st.tabs(["数据集信息", "模型训练与评估", "上传图片预测"])

with tab1:
    st.write("数据集图像形状:", faces.images.shape)

    col1, col2 = st.columns(2)
    with col1:
        sample_index = np.random.choice(len(faces.images))
        sample_image = faces.images[sample_index]
        st.image(sample_image, caption=faces.target_names[faces.target[sample_index]])

    with col2:
        st.write("数据集信息:")
        st.write(f"- 样本数量: {len(faces.images)}")
        st.write(f"- 特征数量: {faces.data.shape[1]}")
        st.write(f"- 类别数量: {len(faces.target_names)}")

with tab2:
    if st.button("加载模型"):
        # 加载模型
        with st.spinner("正在加载模型..."):
            svm_clf, scaler = load_models()
        st.success("模型加载成功！")
    else:
        st.info("点击上方按钮加载模型")

with tab3:
    uploaded_file = st.file_uploader(
        "上传一张人像图片进行预测", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        st.image(image, caption="上传的图片", use_column_width=True)

        if "svm_clf" in locals() and "scaler" in locals():
            target_shape = faces.images[0].shape
            preprocessed_image = preprocess_image(image_array, target_shape)
            preprocessed_image_scaled = scaler.transform([preprocessed_image])
            prediction = svm_clf.predict(preprocessed_image_scaled)
            predicted_person = faces.target_names[prediction[0]]
            st.success(f"预测结果: {predicted_person}")
        else:
            st.error("请先加载模型")
