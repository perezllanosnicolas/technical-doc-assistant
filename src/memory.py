from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from .chains import get_llm

def get_conversational_chain(vectorstore):
    memory = ConversationBufferWindowMemory(
        k=5,                          # Remember last 5
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )