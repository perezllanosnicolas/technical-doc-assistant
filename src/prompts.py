from langchain.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_template("""
You are a technical documentation assistant. Answer the question 
using ONLY the provided context. If the answer is not in the context, 
say "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer (be concise and technical):
""")

REQUIREMENTS_PROMPT = ChatPromptTemplate.from_template("""
Extract all technical requirements from the following document excerpt.
Format each requirement as: REQ-[NUMBER]: [description]
Mark as (FUNCTIONAL) or (NON-FUNCTIONAL).

Document:
{context}

Requirements:
""")

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
Generate a structured executive summary of this technical document.
Use the following format:
## Objective
## Key Technical Specifications  
## Main Constraints
## Open Points

Document:
{context}

Summary:
""")