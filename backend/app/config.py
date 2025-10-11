import os
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, validator

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = Field("demo_key", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: str = Field(None, env="GOOGLE_API_KEY")
    groq_api_key: str = Field(None, env="GROQ_API_KEY")
    
    app_host: str = Field("127.0.0.1", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    debug: bool = Field(True, env="DEBUG")
    
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    max_file_size: int = Field(20971520, env="MAX_FILE_SIZE")
    allowed_extensions: str = Field("pdf,txt,md,docx,xlsx,xls,json,jpg,jpeg,png,bmp,tiff", env="ALLOWED_EXTENSIONS")
    
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    
    default_llm: str = Field("groq", env="DEFAULT_LLM")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-haiku-20240307", env="ANTHROPIC_MODEL")
    google_model: str = Field("gemini-2.5-flash", env="GOOGLE_MODEL")
    groq_model: str = Field("llama-3.1-8b-instant", env="GROQ_MODEL")
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    def get_allowed_extensions(self) -> List[str]:
        """Get allowed extensions as a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    class Config:
        env_file = ".env"

settings = Settings()
