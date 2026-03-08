"""
Module 11 — Logging and Debugging
Exercise 02 — Structured Logging for Benchmarks

WHAT YOU'RE BUILDING:
  Structured logging (JSON format) enables log analysis:
  - Parse logs with tools like jq, grep, or ELK stack
  - Extract metrics, filter by level, aggregate results
  - This is how large-scale systems handle observability.

OBJECTIVE:
  - Create JSON-formatted log entries
  - Include structured context (size, dtype, device)
  - Use logging.Filter for contextual enrichment
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the advantage of JSON logs over text logs?
# Q2: How does a logging.Filter add context to every log entry?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import logging
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkContext:
    """Context for benchmark logging."""
    kernel: str
    dtype: str
    device: str
    size: Optional[int] = None

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Create a JSON formatter for structured logging.
#              Each log entry becomes a JSON object with timestamp, level, message.
# HINT: class JSONFormatter(logging.Formatter): def format(self, record): ...

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured analysis."""
    
    def format(self, record: logging.LogRecord) -> str:
        # TODO: create dict with timestamp, level, logger, message
        #       Convert to JSON string
        pass

# TODO [MEDIUM]: Create a logging.Filter that adds context to every entry.
#              This is how you add kernel/size/device to all logs automatically.
# HINT: class ContextFilter(logging.Filter): def filter(self, record): record.kernel = ...

class BenchmarkContextFilter(logging.Filter):
    """Add benchmark context to every log entry."""
    
    def __init__(self, context: BenchmarkContext):
        self.context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        # TODO: add context fields to record
        #       record.kernel = self.context.kernel, etc.
        pass

# TODO [EASY]: Setup logger with JSON formatter and context filter.
#              This is the full structured logging setup.
# HINT: handler.setFormatter(JSONFormatter()); logger.addFilter(BenchmarkContextFilter(ctx))

def setup_structured_logger(
    name: str,
    context: BenchmarkContext
) -> logging.Logger:
    """Setup logger with JSON formatting and context.
    
    Args:
        name: Logger name
        context: Benchmark context to include in all logs
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # TODO: setup handler with JSON formatter and context filter
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How would you parse JSON logs with jq or Python?
# C2: When is structured logging overkill vs essential?

if __name__ == "__main__":
    print("Testing structured logging...\n")
    
    # Create context
    context = BenchmarkContext(
        kernel="matmul",
        dtype="float16",
        device="cuda:0"
    )
    
    # Setup logger
    logger = setup_structured_logger(__name__, context)
    
    # Log some entries
    logger.info("Benchmark started", extra={"size": 1024})
    logger.info("Benchmark completed", extra={"ms": 0.21, "tflops": 102.4})
    
    print("\nDone! Check console for JSON output")
