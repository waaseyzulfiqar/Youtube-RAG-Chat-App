import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import faiss
import numpy as np
import os
import re

load_dotenv()

def clean_response(text):
    # Remove everything between <think> and </think>, including the tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎥")
st.title("🎥 Chat with a YouTube Video")

# --- Helper: extract video ID from any YouTube URL format ---
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# --- Cache the heavy model so it loads only once ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

# --- Session state to keep the index across reruns ---
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.messages = []

# --- Step 1: URL input ---
url = st.text_input("Paste a YouTube URL:")

if st.button("Load video"):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Couldn't find a valid video ID in that URL.")
    else:
        with st.spinner("Fetching transcript and building index..."):
            try:
                api = YouTubeTranscriptApi()
                transcript = api.fetch(video_id, languages=["en-US", "en", "hi", "bn"])
                full_text = " ".join([snippet.text for snippet in transcript])

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(full_text)

                embeddings = embed_model.encode(chunks)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(embeddings).astype("float32"))

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.messages = []
                st.success(f"Video loaded — {len(chunks)} chunks indexed. Ask away!")
            except Exception as e:
                st.error(f"Couldn't load transcript: {e}")

# --- Step 2: Chat interface ---
if st.session_state.index is not None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the video...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                question_embedding = embed_model.encode([question])
                k = 3
                distances, indices = st.session_state.index.search(
                    np.array(question_embedding).astype("float32"), k
                )
                retrieved_chunks = [st.session_state.chunks[idx] for idx in indices[0]]
                context = "\n\n".join(retrieved_chunks)

                prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

                response = hf_client.chat_completion(
                    model="deepseek-ai/DeepSeek-R1:fastest",
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = answer = clean_response(response.choices[0].message.content)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})