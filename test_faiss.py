from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

video_id = "mlp4IueSJUE"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])
full_text = " ".join([snippet.text for snippet in transcript])

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Build the FAISS index
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Test a search
question = "What is this video about?"
question_embedding = model.encode([question])

k = 3  # top 3 most relevant chunks
distances, indices = index.search(np.array(question_embedding).astype("float32"), k)

print("Top matching chunks:\n")
for idx in indices[0]:
    print(chunks[idx])
    print("---")