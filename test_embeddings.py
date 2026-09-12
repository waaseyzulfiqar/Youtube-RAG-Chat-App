from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

video_id = "mlp4IueSJUE"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])
full_text = " ".join([snippet.text for snippet in transcript])

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print(f"Number of chunks: {len(chunks)}")
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First embedding (first 10 numbers): {embeddings[0][:10]}")