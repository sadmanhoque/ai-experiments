This is meant to be similar to what I was trying to do in the basic-setup directory but there I started without doing much research into LLM functionality and I haven't touched the  code in a while, so this is a fresh start to avoid wasting time cleaning/modifying stuff.


==================================================

THE GOAL:

Use the Retrieval-Augmented Generation (RAG) method of generating answers to create a hyper specific LLMs that is less likely to hallucinate, hopefully, and provide every response with proper citation. Essentially, the LLM shouldn't answer the question, it should find me where in my provided documents, the answer lies. Think of it like an evolution of ctlr+F button.

===================================================

Setup instructions:
(commands are based on running in Linux distro pop_os 22.04)

- download ollama
- Install from https://ollama.com
- ollama pull llama3
- pip install fastapi uvicorn langchain langchain-community langchain-classic chromadb sentence-transformers beautifulsoup4
- python3 ingest.py  (first time only)
- ollama serve (to start the local llm)
- uvicorn server:app --reload  (to start up the webapp at localhost:8000)
- 