# === indexer.py (with namespace support by subject) ===

import os
import re
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone
from collections import Counter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

# === Load environment ===
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = "career-launcher-index"

# === Pinecone setup ===
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# === PDF Loading ===
def load_pdfs_with_metadata(data: str) -> List[Document]:
    all_docs = []
    for subject_folder in os.listdir(data):
        subject_path = os.path.join(data, subject_folder)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(subject_path, filename)
                    loader = PyMuPDFLoader(file_path)
                    try:
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["subject"] = subject_folder
                            doc.metadata["source"] = filename
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")
    return all_docs

# === Chunking ===
def chunk_documents(docs: List[Document]) -> List[Document]:
    chunks = []
    regex_splitter = re.compile(r"(?m)^\s*(\d{1,2})\.\s+")
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    BYTE_LIMIT = 3800000

    for doc in docs:
        text = doc.page_content
        questions = regex_splitter.split(text)
        for i in range(1, len(questions), 2):
            number = questions[i].strip()
            content = questions[i + 1].strip()

            chunk = Document(
                page_content=f"{number}. {content}",
                metadata={**doc.metadata, "question_number": number}
            )

            byte_size = len(chunk.page_content.encode("utf-8"))
            if byte_size > BYTE_LIMIT:
                print(f"⚠️ Chunk too large (Q{number}, {byte_size/1024:.2f} KB), splitting...")
                subchunks = fallback_splitter.split_documents([chunk])
                for sub in subchunks:
                    if len(sub.page_content.encode("utf-8")) <= BYTE_LIMIT:
                        sub.metadata = chunk.metadata.copy()
                        chunks.append(sub)
                    else:
                        print(f"❌ Still too large after split: Skipping Q{number}")
            else:
                chunks.append(chunk)

    return chunks

# === Hash-based Unique ID Generator ===
def generate_id(doc: Document) -> str:
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
    subject = doc.metadata.get("subject", "unknown")
    source = doc.metadata.get("source", "unknown")
    qno = doc.metadata.get("question_number", "0")
    return f"{subject}-{source}-{qno}-{content_hash[:8]}"

# === Load, Chunk, Upload with Namespace ===
docs = load_pdfs_with_metadata("data")
chunks = chunk_documents(docs)
safe_chunks = [c for c in chunks if len(c.page_content.encode("utf-8")) <= 3800000]

# === Print stats ===
print(f"✅ Proceeding with {len(safe_chunks)} chunks after filtering oversized ones.")
q_distribution = Counter([doc.metadata.get("question_number") for doc in safe_chunks])
print("\n📊 Chunks per question number:")
for q, count in sorted(q_distribution.items(), key=lambda x: int(x[0])):
    print(f"  Question {q}: {count} chunk(s)")

# === Upload per namespace ===
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
namespace_groups = {}
for doc in safe_chunks:
    ns = doc.metadata.get("subject", "general")
    namespace_groups.setdefault(ns, []).append(doc)

for ns, group in namespace_groups.items():
    print(f"\n📤 Uploading {len(group)} chunks to namespace: '{ns}'")
    vectorstore = PineconeVectorStore.from_documents(
        documents=group,
        index_name=index_name,
        embedding=embedding_model,
        namespace=ns,
        ids=[generate_id(doc) for doc in group]
    )

print("✅ Indexing complete.")
