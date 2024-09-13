import streamlit as st
from ollama_api import ollama_generator, get_available_models  # 导入 Ollama API 函数
from hunyuan_api import hunyuan_generator  # 导入混元大模型 API 函数

st.set_page_config(
    page_title="chat",
    page_icon="💬",
)
st.sidebar.header("与 AI 聊天")
st.header("聊天")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = "混元"  # 默认平台为混元
if "messages" not in st.session_state:
    st.session_state.messages = []

# 选择使用的平台（混元 或 Ollama）
st.session_state.selected_platform = st.selectbox(
    "请选择平台：",
    ["混元", "Ollama"],
    index=0,  # 默认选择混元
)

# 根据选择的平台获取相应的模型列表
if st.session_state.selected_platform == "Ollama":
    st.session_state.selected_model = st.selectbox(
        "请选择 Ollama 模型：", get_available_models()
    )
else:
    st.session_state.selected_model = st.selectbox(
        "请选择混元模型：",
        [
            "hunyuan-lite",
            "hunyuan-standard",
            "hunyuan-standard-256K",
            "hunyuan-pro",
            "hunyuan-code",
            "hunyuan-role",
            "hunyuan-functioncall",
            "hunyuan-vision",
        ],
    )

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请问有什么可以帮助您？"):
    # 保存用户输入的消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 根据选定的平台生成模型的回复
    with st.chat_message("assistant"):
        if st.session_state.selected_platform == "Ollama":
            response = st.write_stream(
                ollama_generator(
                    st.session_state.selected_model, st.session_state.messages
                )
            )
        else:
            response = st.write_stream(
                hunyuan_generator(
                    st.session_state.selected_model, st.session_state.messages
                )
            )

    st.session_state.messages.append({"role": "assistant", "content": response})

