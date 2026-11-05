"""
Configuration module for loading environment variables
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator

DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

class Settings(BaseSettings):
    #App settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    #PINECONE VECTOR DATABASE
    PINECONE_DB_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = ""

    #HuggingFace settings for embeddings
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_EMBEDDING_MODEL: str = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL

    @field_validator("HUGGINGFACE_EMBEDDING_MODEL", mode="before")
    @classmethod
    def default_empty_embedding_model(cls, value: str | None) -> str:
        if value is None or not value.strip():
            return DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
        return value.strip()

    #groq settings for llm
    GOOGLE_GENERATIVE_AI_API_KEY  : str = ""
    LLM_MODEL: str = "gemini-3.6-flash" #gemini model

    #LangGraph settings
    LANGGRAPH_CHECKPOINTER: str = "memory"

    #File processing settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024 # 50MB
    CHUNK_SIZE: int = 1000 
    CHUNK_OVERLAP: int = 200

    #Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()    

# Validate required settings
def validate_settings():
    """ Validate that required environment variables are set"""
    required_vars = []

    if not settings.GOOGLE_GENERATIVE_AI_API_KEY :
        required_vars.append("GOOGLE_GENERATIVE_AI_API_KEY ")

    if not settings.PINECONE_DB_API_KEY:
        required_vars.append("PINECONE_DB_API_KEY")

    if not settings.PINECONE_ENVIRONMENT:
        required_vars.append("PINECONE_ENVIRONMENT")

    if required_vars:
        raise ValueError(f"Missing required environment varaibles: {', '.join(required_vars)}")

try:
    validate_settings()        
except ValueError as e:
    print(f"Configuration warning {e}")
    print("Please set the requierd environment variables in your .env file")
    print("Required variables: GOOGLE_GENERATIVE_AI_API_KEY, PINECONE_DB_API_KEY, PINECONE_ENVIRONMENT")