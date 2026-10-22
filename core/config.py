"""
Configuration module for loading environment variables
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #App settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    #PINECONE VECTOR DATABASE
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "rag-index"

    #HuggingFace settings for embeddings
    HUGGINGFACE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2" #open-source embedding model dimension - 768

    #groq settings for llm
    GROK_API_KEY : str = ""
    LLM_MODEL: str = "gemma2-9b-it" #groq model

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

    if not settings.GROK_API_KEY:
        required_vars.append("GROQ_API_KEY")

    if not settings.PINECONE_API_KEY:
        required_vars.append("PINECONE_API_KEY")

    if not settings.PINECONE_ENVIRONMENT:
        required_vars.append("PINECONE_ENVIRONMENT")

    if required_vars:
        raise ValueError(f"Missing required environment varaibles: {', '.join(required_vars)}")

try:
    validate_settings()        
except ValueError as e:
    print(f"Configuration warning {e}")
    print("Please set the requierd environment variables in your .env file")
    print("Required variables: GROQ_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT")