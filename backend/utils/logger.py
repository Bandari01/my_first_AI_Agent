"""
Logging Utility
"""
import sys
from pathlib import Path
from loguru import logger
from backend.config import LOG_LEVEL, PROJECT_ROOT

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Add console output
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Add file output
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="00:00",  # Rotate daily
    retention="30 days",  # Keep 30 days
    compression="zip"  # Compress old logs
)

def get_logger(name: str):
    """Get named logger"""
    return logger.bind(name=name)

__all__ = ["logger", "get_logger"]

