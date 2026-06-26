from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from .prompts import QA_PROMPT, REQUIREMENTS_PROMPT, SUMMARY_PROMPT
import os

def get_llm(model: str = "llama3-8b-8192"):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model,
        temperature=0.1   
    )

def get_qa_chain(vectorstore):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

def extract_requirements(vectorstore, llm=None):
    if not llm:
        llm = get_llm()

    docs = vectorstore.similarity_search("requirements specifications constraints", k=6)
    context = "\n\n".join(d.page_content for d in docs)
    chain = REQUIREMENTS_PROMPT | llm
    return chain.invoke({"context": context})

def generate_summary(vectorstore, llm=None):
    if not llm:
        llm = get_llm()
    docs = vectorstore.similarity_search("objective purpose scope specifications", k=6)
    context = "\n\n".join(d.page_content for d in docs)
    chain = SUMMARY_PROMPT | llm
    return chain.invoke({"context": context})