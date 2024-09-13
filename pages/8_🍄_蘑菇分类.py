import streamlit as st
import pandas as pd
import joblib

# 标题
st.title("蘑菇分类预测")

st.image("data/毒蘑菇分类.webp")
st.caption(
    "先对 8124 个蘑菇的样本使用随机森林分类器进行训练，现在使用训练好的模型预测蘑菇是否有毒。请选择以下特征来进行预测。"
)


# 加载模型和特征列
@st.cache_resource
def load_model_and_columns(model_path, columns_path):
    """加载模型和特征列"""
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    return model, columns


model, feature_columns = load_model_and_columns(
    "data/mushrooms.pkl", "data/feature_columns.pkl"
)

# 蘑菇特征选择器
cap_shape = st.selectbox(
    "菌盖形状", ["bell", "conical", "convex", "flat", "knobbed", "sunken"]
)
cap_surface = st.selectbox("菌盖表面", ["fibrous", "grooves", "scaly", "smooth"])
cap_color = st.selectbox(
    "菌盖颜色",
    [
        "brown",
        "buff",
        "cinnamon",
        "gray",
        "green",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    ],
)
bruises = st.selectbox("是否有瘀伤", ["bruises", "no"])
odor = st.selectbox(
    "气味",
    [
        "almond",
        "anise",
        "creosote",
        "fishy",
        "foul",
        "musty",
        "none",
        "pungent",
        "spicy",
    ],
)
gill_attachment = st.selectbox("鳃附着", ["attached", "descending", "free", "notched"])
gill_spacing = st.selectbox("鳃间距", ["close", "crowded", "distant"])
gill_size = st.selectbox("鳃大小", ["broad", "narrow"])
gill_color = st.selectbox(
    "鳃颜色",
    [
        "black",
        "brown",
        "buff",
        "chocolate",
        "gray",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    ],
)
stalk_shape = st.selectbox("茎形状", ["enlarging", "tapering"])
stalk_root = st.selectbox(
    "茎根", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"]
)
stalk_surface_above_ring = st.selectbox(
    "环上茎表面", ["fibrous", "scaly", "silky", "smooth"]
)
stalk_surface_below_ring = st.selectbox(
    "环下茎表面", ["fibrous", "scaly", "silky", "smooth"]
)
stalk_color_above_ring = st.selectbox(
    "环上茎颜色",
    ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"],
)
stalk_color_below_ring = st.selectbox(
    "环下茎颜色",
    ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"],
)
veil_type = st.selectbox("幕类型", ["partial"])
veil_color = st.selectbox("幕颜色", ["brown", "orange", "white", "yellow"])
ring_number = st.selectbox("环数", ["none", "one", "two"])
ring_type = st.selectbox(
    "环类型",
    [
        "cobwebby",
        "evanescent",
        "flaring",
        "large",
        "none",
        "pendant",
        "sheathing",
        "zone",
    ],
)
spore_print_color = st.selectbox(
    "孢子印颜色",
    [
        "black",
        "brown",
        "buff",
        "chocolate",
        "green",
        "orange",
        "purple",
        "white",
        "yellow",
    ],
)
population = st.selectbox(
    "种群", ["abundant", "clustered", "numerous", "scattered", "several", "solitary"]
)
habitat = st.selectbox(
    "栖息地", ["grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"]
)

# 编码特征
input_features = {
    "cap-shape": cap_shape,
    "cap-surface": cap_surface,
    "cap-color": cap_color,
    "bruises": bruises,
    "odor": odor,
    "gill-attachment": gill_attachment,
    "gill-spacing": gill_spacing,
    "gill-size": gill_size,
    "gill-color": gill_color,
    "stalk-shape": stalk_shape,
    "stalk-root": stalk_root,
    "stalk-surface-above-ring": stalk_surface_above_ring,
    "stalk-surface-below-ring": stalk_surface_below_ring,
    "stalk-color-above-ring": stalk_color_above_ring,
    "stalk-color-below-ring": stalk_color_below_ring,
    "veil-type": veil_type,
    "veil-color": veil_color,
    "ring-number": ring_number,
    "ring-type": ring_type,
    "spore-print-color": spore_print_color,
    "population": population,
    "habitat": habitat,
}

# 将输入的特征转换为独热编码
input_df = pd.DataFrame([input_features])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# 进行预测
if st.button("预测"):
    prediction = model.predict(input_encoded)[0]
    prediction_label = "可食用" if prediction == "e" else "有毒"

    # 显示预测结果
    st.subheader("预测结果")
    st.write(f"该蘑菇被预测为: **{prediction_label}**")

    # 显示输入的特征值
    st.subheader("输入的特征值")
    st.write(input_features)
