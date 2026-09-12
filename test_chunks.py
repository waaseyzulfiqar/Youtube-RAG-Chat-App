from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

video_id = "mlp4IueSJUE"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])
full_text = " ".join([snippet.text for snippet in transcript])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(full_text)

print(f"Total chunks: {len(chunks)}")
print("First chunk:\n", chunks[0])