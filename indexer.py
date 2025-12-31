# indexer.py

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
index_name = "career-launcher"

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
    section_pattern = r"(?m)(^\s*(Diagnostic Exercise\s*[-\u2013]\s*\d+|Exercise\s*[-\u2013]\s*\d+)\s*$)"
    question_splitter = re.compile(r"(?m)^\s*(\d{1,2})\.\s+")
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    BYTE_LIMIT = 3800000

    if not hasattr(chunk_documents, "current_exercise"):
        chunk_documents.current_exercise = "Unknown Section"
    if not hasattr(chunk_documents, "current_passage"):
        chunk_documents.current_passage = ""

    for doc in docs:
        text = doc.page_content
        parts = re.split(section_pattern, text)
        parts = [p.strip() for p in parts if p.strip()]

        i = 0
        while i < len(parts):
            part = parts[i]
            if re.match(r"(?i)Diagnostic Exercise\s*[-\u2013]\s*\d+|Exercise\s*[-\u2013]\s*\d+", part):
                chunk_documents.current_exercise = part
                chunk_documents.current_passage = ""
                i += 1
            if i < len(parts):
                content = parts[i]
                questions = question_splitter.split(content)
                local_passage = re.sub(r'(?m)^\s*(Page \d+\s*RC - 01\s*|\s*RC - 01\s*Page \d+\s*)', '', questions[0]).strip()
                if local_passage:
                    chunk_documents.current_passage = local_passage
                for j in range(1, len(questions), 2):
                    number = questions[j].strip()
                    qcontent = re.sub(r'(?m)^\s*(Page \d+\s*RC - 01\s*|\s*RC - 01\s*Page \d+\s*)', '', questions[j + 1]).strip()
                    qcontent = re.sub(r'\nYour Score .*$', '', qcontent, flags=re.M)
                    if number and qcontent:
                        full_content = (chunk_documents.current_passage + "\n\n" if chunk_documents.current_passage else "") + f"{number}. {qcontent}"
                        chunk = Document(
                            page_content=full_content,
                            metadata={**doc.metadata, "question_number": number, "exercise": chunk_documents.current_exercise}
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
                i += 1

    return chunks

# === Hash-based Unique ID Generator ===
def generate_id(doc: Document) -> str:
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
    subject = doc.metadata.get("subject", "unknown")
    source = doc.metadata.get("source", "unknown")
    qno = doc.metadata.get("question_number", "0")
    return f"{subject}-{source}-{qno}-{content_hash[:8]}"

# === Load, Chunk, and Filter ===
docs = load_pdfs_with_metadata("data")
chunks = chunk_documents(docs)
safe_chunks = [c for c in chunks if len(c.page_content.encode("utf-8")) <= 3800000]

print(f"✅ Proceeding with {len(safe_chunks)} chunks after filtering oversized ones.")
q_distribution = Counter([doc.metadata.get("question_number") for doc in safe_chunks])
print("\n📊 Chunks per question number:")
for q, count in sorted(q_distribution.items(), key=lambda x: int(x[0])):
    print(f"  Question {q}: {count} chunk(s)")

# === Upload per namespace (only new chunks) ===
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
namespace_groups = {}
for doc in safe_chunks:
    ns = doc.metadata.get("subject", "general")
    namespace_groups.setdefault(ns, []).append(doc)

for ns, group in namespace_groups.items():
    ids = [generate_id(doc) for doc in group]

    # Fetch existing IDs in Pinecone (batched)
    existing = set()
    try:
        BATCH = 100
        for i in range(0, len(ids), BATCH):
            batch_ids = ids[i:i + BATCH]
            response = index.fetch(ids=batch_ids, namespace=ns)
            existing.update(response.vectors.keys())
    except Exception as e:
        print(f"⚠️ Error checking existing vectors in namespace '{ns}': {e}")

    # Filter out already indexed chunks
    new_docs = [doc for doc, _id in zip(group, ids) if _id not in existing]
    new_ids = [generate_id(doc) for doc in new_docs]

    print(f"\n📤 Uploading {len(new_docs)} new chunks to namespace: '{ns}'")
    if new_docs:
        PineconeVectorStore.from_documents(
            documents=new_docs,
            index_name=index_name,
            embedding=embedding_model,
            namespace=ns,
            ids=new_ids
        )

print("✅ Indexing complete (skipping existing chunks).")