"""
Simple RAG Demo — OpenAI + ChromaDB
Run: python3 app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

# -------------------------------
# STEP 1: Initialize Clients
# -------------------------------
print("🚀 Initializing OpenAI and ChromaDB...")

client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="company_docs")

print("✅ ChromaDB collection ready\n")


# -------------------------------
# STEP 2: Embedding Function
# -------------------------------
def get_embedding(text):
    print(f"🔄 Creating embedding for: '{text[:60]}...'")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding
    print(f"✅ Embedding created (length: {len(embedding)})\n")
    return embedding


# -------------------------------
# STEP 3: Store Documents
# -------------------------------
def store_documents(documents):
    print("📦 Storing documents in ChromaDB...\n")
    logs = []
    chunks_info = []

    for i, doc in enumerate(documents):
        doc_id = f"doc_{collection.count() + i}"
        embedding = get_embedding(doc)

        collection.add(
            documents=[doc],
            embeddings=[embedding],
            ids=[doc_id]
        )

        log = f"✅ Stored Document '{doc_id}' — {doc[:50]}..."
        print(log)
        logs.append(log)
        
        # Save info for the UI to visualize
        chunks_info.append({
            "id": doc_id,
            "text": doc,
            "vector_preview": [round(x, 4) for x in embedding[:5]]
        })

    print(f"\n🎉 All {len(documents)} documents stored! Total in DB: {collection.count()}\n")
    logs.append(f"🎉 Total documents in DB: {collection.count()}")
    return logs, chunks_info


# -------------------------------
# STEP 4: Similarity Search
# -------------------------------
def search_similar(query, top_k=3):
    print(f"❓ User Query: {query}\n")

    query_embedding = get_embedding(query)

    print("🔍 Searching for similar documents...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]

    print("📌 Retrieved Documents:")
    for doc in retrieved_docs:
        print(f"   - {doc[:80]}...")
    print()

    return retrieved_docs, distances


# -------------------------------
# STEP 5: Generate Answer (RAG)
# -------------------------------
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)

    prompt = f"""Answer the question using the context below:

Context:
{context}

Question:
{query}"""

    print("🧾 Final Prompt Sent to LLM:")
    print(prompt)
    print()

    print("🤖 Generating final answer...\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print(f"💡 Final Answer: {answer}\n")
    print("🎯 RAG Pipeline Completed Successfully!\n")
    return answer


# ── API Routes ──────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    text = request.json["text"]

    # Split into sentences/paragraphs as documents
    documents = [line.strip() for line in text.split("\n") if line.strip()]

    print(f"\n📄 Received {len(documents)} documents to ingest")
    logs, chunks_info = store_documents(documents)

    return jsonify({
        "logs": logs, 
        "total": collection.count(),
        "chunks_info": chunks_info
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    query = request.json["query"]

    if collection.count() == 0:
        return jsonify({"error": "No documents yet! Ingest some first."}), 400

    # STEP 4: Search
    retrieved_docs, distances = search_similar(query)
    scores = [round(1 - d, 4) for d in distances]

    # STEP 5: Generate
    answer = generate_answer(query, retrieved_docs)

    logs = [
        f"🔍 Retrieved {len(retrieved_docs)} chunks (scores: {scores})",
        f"🤖 Generated answer from GPT-4o-mini",
        f"🎯 RAG Pipeline Complete!"
    ]

    return jsonify({
        "answer": answer,
        "chunks": retrieved_docs,
        "scores": scores,
        "logs": logs
    })


if __name__ == "__main__":
    print("\n⚡ RAG Demo running at http://localhost:5050\n")
    app.run(port=5050, debug=True)
