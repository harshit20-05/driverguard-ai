"""
Logging infrastructure for DriverGuard AI.
Configures file and console loggers with formatted output.
"""

import os
import sys
import logging
from typing import Optional


_ROOT_LOGGER_CONFIGURED = False


def setup_logger(
    name: str = "driverguard",
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Initializes and configures a logger with both console and rotating file output.
    
    Args:
        name: Name of the logger
        log_dir: Directory where log file should be stored
        log_file: File name for logs
        level: Minimum logging level
        
    Returns:
        Configured logging.Logger instance
    """
    global _ROOT_LOGGER_CONFIGURED
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to attach file log handler to {log_filepath}: {e}\n")
        
    _ROOT_LOGGER_CONFIGURED = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a child or root logger instance.
    """
    base_name = "driverguard"
    if name:
        target_name = f"{base_name}.{name}"
    else:
        target_name = base_name
        
    logger = logging.getLogger(target_name)
    if not logger.handlers and not logging.getLogger(base_name).handlers:
        setup_logger(base_name)
    return logger
