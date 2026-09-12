from youtube_transcript_api import YouTubeTranscriptApi

video_id = "mlp4IueSJUE"

api = YouTubeTranscriptApi()
transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])

full_text = " ".join([snippet.text for snippet in transcript])
print(full_text[:500])