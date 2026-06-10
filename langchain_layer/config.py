from pydantic_settings import BaseSettings

class LangChainConfig(BaseSettings):
    MAX_HISTORY_TOKENS: int = 2000
    MEMORY_KEY: str = "chat_history"
    SESSION_TTL: int = 3600
    
    class Config:
        env_prefix = "LANGCHAIN_"

langchain_config = LangChainConfig()
