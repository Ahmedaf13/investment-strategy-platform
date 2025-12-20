import sys
from pathlib import Path
from loguru import logger


def setup_logging(level="INFO", log_file=None, rotation="10 MB"):
    logger.remove()
    
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    )
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation=rotation, retention="7 days")
    
    logger.info(f"Logging configured: level={level}")


def format_currency(value, currency="$"):
    if abs(value) >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    return f"{currency}{value:.2f}"


def format_percentage(value, decimals=2):
    return f"{value * 100:.{decimals}f}%"
