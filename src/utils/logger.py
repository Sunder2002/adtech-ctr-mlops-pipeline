import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """Configures a production-grade logger with console and rotating file handlers."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # 1. Console Handler (for local terminal debugging)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (for production persistent logging)
        # Keeps up to 5 log files of 5MB each, rotating them automatically
        os.makedirs("logs", exist_ok=True)
        file_handler = RotatingFileHandler(
            "logs/mlops_pipeline.log", maxBytes=5*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger