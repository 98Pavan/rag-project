import math
from google import genai

# =====================
# SETUP
# =====================
API_KEY = "AIzaSyCiRwUf9EGmIlobdAih5Q5y5umvlPFX31k"
client = genai.Client(api_key=API_KEY)

# =====================
# STEP 2: LOAD DOCUMENT
# =====================
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# =====================
# STEP 3: CHUNK TEXT
# =====================
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# =====================
# STEP 4: GET EMBEDDINGS
# =====================
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

# =====================
# STEP 5: VECTOR STORE
# =====================
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    return dot_product / (magnitude_a * magnitude_b)

def search_chunks(query, embedded_chunks, top_k=2):
    query_embedding = get_embedding(query)
    similarities = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        similarities.append({
            "text": chunk["text"],
            "score": score
        })
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]

# =====================
# STEP 6: GENERATION
# =====================
def generate_answer(question, relevant_chunks):
    context = "\n".join([chunk["text"] for chunk in relevant_chunks])
    prompt = f"""You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text