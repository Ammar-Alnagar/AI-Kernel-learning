"""
Module 11 — Logging and Debugging
Exercise 01 — Python Logging Basics

WHAT YOU'RE BUILDING:
  Production kernel code uses logging, not print():
  - Structured logs with levels (DEBUG, INFO, WARNING, ERROR)
  - Timestamps, loggers, handlers
  - File output for benchmark results
  
  This is how PyTorch, DeepSpeed, and vLLM handle observability.

OBJECTIVE:
  - Configure logging with handlers and formatters
  - Use appropriate log levels
  - Log to file and console simultaneously
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between logging.info() and print()?
# Q2: Why use getLogger(__name__) instead of root logger?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import logging
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Configure a logger with console handler.
#              Set level, formatter, and add handler.
# HINT: logger = logging.getLogger(__name__); logger.setLevel(logging.DEBUG)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with console handler.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # TODO: create console handler, set formatter, add to logger
    # Formatter: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    pass

# TODO [EASY]: Add file handler to existing logger.
#              This is how you persist benchmark logs.
# HINT: file_handler = logging.FileHandler("benchmark.log")

def add_file_handler(logger: logging.Logger, log_path: str) -> logging.FileHandler:
    """Add file handler to logger.
    
    Args:
        logger: Existing logger
        log_path: Path to log file
    
    Returns:
        File handler (for later removal if needed)
    """
    # TODO: implement file handler setup
    pass

# TODO [MEDIUM]: Log benchmark progress at appropriate levels.
#              DEBUG for details, INFO for progress, ERROR for failures.
# HINT: logger.debug(), logger.info(), logger.error()

def log_benchmark_run(logger: logging.Logger, size: int, result: Optional[float]):
    """Log benchmark run with appropriate levels."""
    # TODO: implement logging at different levels
    # - DEBUG: Starting benchmark for size X
    # - INFO: Completed benchmark for size X: Y ms
    # - ERROR: Failed benchmark for size X
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What log level should you use for routine progress vs debugging details?
# C2: Why is structured logging better than print for production code?

if __name__ == "__main__":
    print("Testing logging basics...\n")
    
    # Setup logger
    logger = setup_logger(__name__, level=logging.DEBUG)
    
    # Test console logging
    logger.debug("Debug message (detailed info)")
    logger.info("Info message (routine progress)")
    logger.warning("Warning message (something unexpected)")
    logger.error("Error message (something failed)")
    
    # Test file logging
    log_path = "test_benchmark.log"
    add_file_handler(logger, log_path)
    
    logger.info(f"Logging to file: {log_path}")
    
    # Test benchmark logging
    log_benchmark_run(logger, 1024, 0.21)
    log_benchmark_run(logger, 2048, None)  # Simulated failure
    
    print(f"\nCheck {log_path} for file output")
    print("\nDone!")
