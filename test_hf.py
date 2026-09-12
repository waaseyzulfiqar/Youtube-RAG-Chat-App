from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

response = client.chat_completion(
    model="deepseek-ai/DeepSeek-R1:fastest",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50
)

print(response.choices[0].message.content)