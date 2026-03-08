"""
Module 10 — Error Handling
Exercise 03 — Context Managers for Error Handling

WHAT YOU'RE BUILDING:
  Context managers centralize error handling patterns:
  - Automatic retry on transient failures
  - Timeout enforcement for hanging operations
  - Resource cleanup on any exit path
  
  This is how production systems handle reliability.

OBJECTIVE:
  - Build context managers that handle errors
  - Implement timeout with signal/alarm or threading
  - Use contextlib for simpler context managers
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How does a context manager's __exit__ handle exceptions?
# Q2: What does returning True from __exit__ do?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import time
import signal
from contextlib import contextmanager
from typing import Optional, Type, Any

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Implement a timeout context manager.
#              Use signal.alarm on Unix or threading Timer.
#              Raise TimeoutError if operation takes too long.
# HINT: signal.signal(signal.SIGALRM, handler); signal.alarm(seconds)

class TimeoutError(Exception):
    """Raised when operation exceeds timeout."""
    pass

@contextmanager
def timeout_context(seconds: int):
    """Context manager that times out after N seconds.
    
    Usage:
        with timeout_context(5):
            long_running_operation()
    """
    # TODO: implement timeout with signal.alarm
    # Note: signal only works on Unix, main thread
    pass

# TODO [MEDIUM]: Implement retry context manager.
#              Retry on specific exceptions with backoff.
# HINT: Use @contextmanager, try/except loop, yield

@contextmanager
def retry_context(
    max_retries: int = 3,
    retry_exceptions: tuple = (Exception,),
    base_delay: float = 0.1
):
    """Context manager that retries on failure.
    
    Usage:
        with retry_context(max_retries=3):
            flaky_operation()
    """
    # TODO: implement retry logic
    pass

# TODO [HARD]: Combine timeout + retry in a single context manager.
#              This is a production-ready pattern.
# HINT: Nest the contexts or implement combined logic

@contextmanager
def reliable_operation_context(
    timeout_seconds: int = 30,
    max_retries: int = 3,
    retry_exceptions: tuple = (ConnectionError, TimeoutError)
):
    """Context manager with timeout and retry.
    
    Usage:
        with reliable_operation_context():
            download_weights()
    """
    # TODO: implement combined timeout + retry
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What are the limitations of signal.alarm for timeout?
# C2: When would you use a context manager vs decorator for retry?

if __name__ == "__main__":
    import platform
    
    print("Testing error handling context managers...\n")
    
    # Test timeout (Unix only)
    if platform.system() != "Windows":
        print("Testing timeout_context...")
        try:
            with timeout_context(1):
                time.sleep(2)  # Should timeout
        except TimeoutError as e:
            print(f"  Timeout caught: {e}")
        except Exception as e:
            print(f"  Other error (expected on some systems): {e}")
    else:
        print("Skipping timeout test (Windows doesn't support signal.alarm)")
    
    # Test retry
    print("\nTesting retry_context...")
    attempt_count = 0
    
    def flaky_op():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Transient failure")
        return "Success"
    
    try:
        # Note: retry_context as written needs adjustment for this usage
        # This is a learning exercise
        result = flaky_op()
        print(f"  Result: {result} after {attempt_count} attempts")
    except ConnectionError:
        print(f"  Failed after {attempt_count} attempts")
    
    print("\nDone!")
