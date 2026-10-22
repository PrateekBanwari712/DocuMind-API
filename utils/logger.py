"""
Centralized logging configuration
"""

import logging
import sys
from typing import Optional
from core.config import settings

#Configuring logging format
formatter = logging.Formatter(settings.LOG_FORMAT)

# Creating Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

def get_logger(name: str, level: Optional[str] = None ) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name: Logger name (usually __name__)
        level: Log level override

    Returns:
        Configured logger instance    
    """
    logger = logging.getLogger(name)

    #Set level
    log_level = level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    #Add handler if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger    


def setup_logging():
    """
    Setup application-wide logging configuration
    """
    # Setting root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Adding console handler to root logger if not present
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)

    # Suppress some verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logger = get_logger(__name__)
    logger.info("Logging configured successfully")

# Auto setup logging when module is imported
setup_logging()    