from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import faiss
import numpy as np
import os

load_dotenv()

video_id = "mlp4IueSJUE"

# 1. Fetch transcript
api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])
full_text = " ".join([snippet.text for snippet in transcript])

# 2. Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

# 3. Embed chunks
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)

# 4. Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# 5. Ask a question
question = "What is this video about?"
question_embedding = embed_model.encode([question])
k = 3
distances, indices = index.search(np.array(question_embedding).astype("float32"), k)
retrieved_chunks = [chunks[idx] for idx in indices[0]]
context = "\n\n".join(retrieved_chunks)

# 6. Send context + question to the LLM
client = InferenceClient(token=os.getenv("HF_TOKEN"))

prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

response = client.chat_completion(
    model="deepseek-ai/DeepSeek-R1:fastest",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300
)

print(response.choices[0].message.content)