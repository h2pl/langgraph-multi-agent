import os
from dotenv import load_dotenv

load_dotenv()

# 确保 HuggingFace 镜像源在最开始就被设置在环境变量中，防止其他库提前加载导致失效
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


class Config:
    # LLM 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # 向量存储配置
    USE_VECTOR_STORE: bool = os.getenv("USE_VECTOR_STORE", "true").lower() == "true"
    USE_LLM: bool = os.getenv("USE_LLM", "true").lower() == "true"

    # 模拟配置
    SIMULATION_SPEED: int = int(os.getenv("SIMULATION_SPEED", "1"))
    MAX_STEPS_PER_DAY: int = int(os.getenv("MAX_STEPS_PER_DAY", "12"))

    # 记忆配置
    MAX_SHORT_TERM_MEMORY: int = 20
    REFLECTION_THRESHOLD: int = 5  # 每积累多少条记忆触发反思
    IMPORTANCE_THRESHOLD: float = 6.0  # 重要性阈值

    # Web 配置
    WEB_HOST: str = "127.0.0.1"
    WEB_PORT: int = 8001


config = Config()
