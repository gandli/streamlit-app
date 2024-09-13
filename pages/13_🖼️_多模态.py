import streamlit as st
from tokenfree_api import describe_image
import os

st.set_page_config(
    page_title="多模态",
    page_icon="🖼️",
)
st.header("多模态")

uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None:
    # 将上传的文件保存到临时位置
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    # 显示上传的图像
    st.image(uploaded_file, caption="上传的图片", use_column_width=False, width=300)  
    custom_text = st.text_area("输入自定义描述文本", "分析这张图片，判断收货地点是否在店内。请注意商店的常见特征如货架、商品陈列、购物车、收银台等，然后直接回答‘收货地点在店内’或‘收货地点不在店内’。")

    if st.button("获取描述"):
        with st.spinner("描述生成中..."):
            descriptions = describe_image("temp_image.jpg", custom_text)
            full_description = "".join(descriptions)
            
            # 显示生成的描述
            st.markdown("### 图片描述")
            st.write(full_description)
            
            # 删除临时文件
            os.remove("temp_image.jpg")