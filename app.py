import os
import fitz
from flask import Flask, request, jsonify, send_from_directory
from rag import chunk_text, get_embedding, search_chunks, generate_answer

app = Flask(__name__)

# Global list — stays in memory
embedded_chunks = []

@app.route('/')
def home():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)), 'index.html'
    )

@app.route('/ingest', methods=['POST'])
def ingest():
    global embedded_chunks
    embedded_chunks = []
    data = request.json
    text = data.get('text', '')
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embedded_chunks.append({
            "id": i,
            "text": chunk,
            "embedding": embedding
        })
    print(f"✅ Text ingested: {len(embedded_chunks)} chunks")
    return jsonify({"status": "success", "chunks": len(embedded_chunks)})

@app.route('/ingest-pdf', methods=['POST'])
def ingest_pdf():
    global embedded_chunks
    embedded_chunks = []

    try:
        pdf_file = request.files['pdf']
        pdf_bytes = pdf_file.read()

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        print(f"📄 PDF text extracted: {len(text)} characters")

        if not text.strip():
            return jsonify({"status": "error", "message": "No text found"})

        chunks = chunk_text(text, chunk_size=200, overlap=30)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            embedded_chunks.append({
                "id": i,
                "text": chunk,
                "embedding": embedding
            })

        print(f"✅ PDF ingested: {len(embedded_chunks)} chunks")
        return jsonify({
            "status": "success",
            "chunks": len(embedded_chunks),
            "message": f"Successfully ingested {len(embedded_chunks)} chunks"
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global embedded_chunks
    print(f"🔍 Ask called. Chunks available: {len(embedded_chunks)}")

    data = request.json
    question = data.get('question', '')

    if not embedded_chunks:
        return jsonify({"answer": "Please ingest a document first!"})

    relevant_chunks = search_chunks(question, embedded_chunks)
    answer = generate_answer(question, relevant_chunks)
    print(f"✅ Answer generated!")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=False)