import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)

def build_vectorstore(pdf_paths: list[str], persist_dir: str = "vectorstore/") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = []
    for path in pdf_paths:
        text = load_pdf(path)
        chunks = splitter.create_documents(
            [text], 
            metadatas=[{"source": path}]
        )
        docs.extend(chunks)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

def load_vectorstore(persist_dir: str = "vectorstore/") -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)