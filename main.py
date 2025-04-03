import os
import fitz  # PyMuPDF
import pickle  # Để lưu và tải FAISS vector_store
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()

# API Key của HuggingFaceHub
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Thư mục lưu trữ PDF và FAISS
DATA_DIR = "data"
FAISS_PATH = os.path.join(DATA_DIR, "vector_store.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# Hàm lưu file PDF
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Đọc nội dung từ PDF với kiểm tra lỗi
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text("text").strip() for page in doc])
        return text
    except Exception as e:
        print(f"Lỗi khi đọc PDF: {e}")
        return ""

# Tạo vector embeddings và lưu vào FAISS
def create_vector_store(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    documents = [Document(page_content=t) for t in texts]

    # Sử dụng mô hình embeddings hỗ trợ tiếng Việt
    embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Nếu FAISS đã tồn tại, tải lên thay vì tạo lại
    if os.path.exists(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_documents(documents, embeddings)
        with open(FAISS_PATH, "wb") as f:
            pickle.dump(vector_store, f)

    return vector_store

# Tải LLM hỗ trợ tiếng Việt từ HuggingFaceHub
def load_llm():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.1, "max_length": 1024},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

# Truy vấn tài liệu từ FAISS với giới hạn số lượng
def retrieve_docs(vector_store, query, top_k=4):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return retriever.get_relevant_documents(query)

# Hỏi đáp với PDF bằng tiếng Việt
def question_pdf(query, vector_store):
    llm = load_llm()
    docs = retrieve_docs(vector_store, query)

    if not docs:
        return "Không tìm thấy thông tin phù hợp."

    context = "\n\n".join([doc.page_content for doc in docs])

    template = """
    Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên nội dung tài liệu.

    - Hãy trả lời ngắn gọn, súc tích.
    - Nếu có thể, hãy liệt kê thông tin dưới dạng gạch đầu dòng.

    Câu hỏi: {question}
    Ngữ cảnh: {context}

    Câu trả lời:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    return chain.invoke({"question": query, "context": context})
