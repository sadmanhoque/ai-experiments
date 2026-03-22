from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Load local LLM via Ollama
llm = Ollama(model="llama3")

# Prompt that enforces staying within your data
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {question}
Answer:""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    result = qa_chain.invoke(query.message)
    sources = list({doc.metadata["source"] for doc in result["source_documents"]})
    return {
        "answer": result["result"],
        "sources": sources
    }

app.mount("/", StaticFiles(directory=".", html=True), name="static")