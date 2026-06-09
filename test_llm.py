import os
import sys
import httpx
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_API_BASE", "")
model = os.getenv("MODEL_NAME", "gemini-1.5-flash")

print(f"API Key length: {len(api_key)}")
print(f"Base URL: {base_url}")
print(f"Model: {model}")

print("\n1. Testing HTTP connection to Gemini endpoint via httpx...")
try:
    # 测试连接
    url = f"{base_url.rstrip('/')}/models" if base_url else "https://generativelanguage.googleapis.com/v1beta/openai/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(url, headers=headers, timeout=10.0)
    print(f"HTTP Status: {resp.status_code}")
    print(f"HTTP Response: {resp.text[:500]}")
except Exception as e:
    print(f"HTTP Error: {e}")

print("\n2. Testing LangChain ChatOpenAI call...")
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.8,
        "max_tokens": 100,
        "timeout": 10.0
    }
    if base_url:
        kwargs["base_url"] = base_url
    
    llm = ChatOpenAI(**kwargs)
    messages = [HumanMessage(content="Say Hello")]
    res = llm.invoke(messages)
    print(f"LLM Response: {res.content}")
except Exception as e:
    print(f"LLM Error: {e}")
