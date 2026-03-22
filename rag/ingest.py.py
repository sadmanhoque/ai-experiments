from bs4 import BeautifulSoup
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def extract_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

# Load all HTML files
docs = []
for path in Path("html_files").glob("**/*.html"):
    text = extract_text(path)
    if text.strip():
        docs.append({"text": text, "source": str(path)})

print(f"Found {len(docs)} HTML files")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(
    [d["text"] for d in docs],
    metadatas=[{"source": d["source"]} for d in docs]
)

# Embed & store locally (uses a local model, no API key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Indexed {len(chunks)} chunks into ./chroma_db")