import pdfplumber
import re
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def clean_text(text: str) -> str:
    """
    Fix PDF extraction artifacts from academic papers with custom font encoding.
    Handles: missing spaces, hyphenated line-breaks, excessive newlines.
    """
    text = re.sub(r'-\n', '', text)

    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    text = re.sub(r'(?<=[a-zA-Z])(?=\()', ' ', text)

    text = re.sub(r'(?<=\))(?=[a-zA-Z])', ' ', text)

    text = re.sub(r'(?<=[a-z])(?=[0-9])', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        # sort=True preserves reading order (left-to-right, top-to-bottom)
        pages.append(page.get_text("text", sort=True))
    doc.close()
    raw = "\n\n".join(pages)
    return clean_text(raw)

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