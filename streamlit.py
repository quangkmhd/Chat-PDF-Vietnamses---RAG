import streamlit as st
import main  # Import các hàm xử lý PDF từ main.py

st.set_page_config(page_title="Chat với PDF tiếng Việt", layout="wide")

st.title("📄 Chat với tài liệu PDF tiếng Việt")

# Tải lên file PDF
upload_file = st.file_uploader("📂 Tải lên file PDF", type="pdf")

vector_store = None

if upload_file:
    with st.spinner("🔄 Đang xử lý PDF..."):
        file_path = main.save_uploaded_file(upload_file)
        vector_store = main.create_vector_store(file_path)

    if vector_store:
        st.success("✅ PDF đã tải lên và được xử lý thành công!")
    else:
        st.error("⚠️ Không thể tạo vector store từ PDF!")

# Hộp chat để đặt câu hỏi
question = st.text_input("💬 Nhập câu hỏi về PDF:")

if question and vector_store:
    st.chat_message("user").write(question)

    with st.spinner("🤖 Đang tạo phản hồi..."):
        answer = main.question_pdf(question, vector_store)

    st.chat_message("assistant").write(answer)
elif question:
    st.error("⚠️ Hãy tải lên một PDF trước khi đặt câu hỏi!")