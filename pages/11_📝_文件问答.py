import streamlit as st
from hunyuan_api import hunyuan_generator
from docx import Document

st.set_page_config(
    page_title="文件问答",
    page_icon="📝",
)

st.title("📝 文件问答")
st.sidebar.header("与文件交流")
uploaded_file = st.file_uploader("上传文章", type=("txt", "md", "docx"))

question = st.text_input(
    "询问有关文档的内容",
    placeholder="您能给我一个简短的总结吗？",
    disabled=not uploaded_file,
)


def read_docx(file) -> str:
    """读取 .docx 文件的内容"""
    document = Document(file)
    return "\n".join([para.text for para in document.paragraphs])


if uploaded_file and question:
    if uploaded_file.name.endswith(".txt") or uploaded_file.name.endswith(".md"):
        try:
            # 尝试以 UTF-8 编码读取文件
            article = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            # 如果 UTF-8 解码失败，尝试其他常见编码
            try:
                article = uploaded_file.read().decode("gbk")
            except UnicodeDecodeError:
                st.error("解码文件失败。请上传 UTF-8 或 GBK 编码的文本文件。")
                st.stop()
    elif uploaded_file.name.endswith(".docx"):
        # 处理 .docx 文件
        article = read_docx(uploaded_file)
    else:
        st.error("不支持的文件格式。请上传 .txt、.md 或 .docx 文件。")
        st.stop()

    conversation = [
        {"role": "system", "content": "你是一个智能助手。"},
        {"role": "user", "content": f"下面是一个文档:\n\n{article}\n\n{question}"},
    ]

    # 在 Streamlit 界面中创建一个可变的占位符
    answer_placeholder = st.empty()

    # 使用 hunyuan_generator 生成逐步回答
    response_generator = hunyuan_generator("hunyuan-lite", conversation)

    full_response = ""  # 用于拼接完整的回答
    for partial_response in response_generator:
        full_response += partial_response
        answer_placeholder.markdown(full_response)  # 实时更新回答，使用 markdown 显示
