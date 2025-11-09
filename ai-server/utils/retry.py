import asyncio
from typing import Optional
from functools import wraps

from config import stt as stt_config
from .logging import get_logger


logger = get_logger(__name__)


def retry_on_failure(max_retries: Optional[int] = None, delay: Optional[float] = None):
    """Decorator to retry async functions on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts (uses config default if None)
        delay: Initial delay between retries in seconds (uses config default if None)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = max_retries if max_retries is not None else stt_config.MAX_RETRIES
            retry_delay = delay if delay is not None else stt_config.RETRY_DELAY_SECONDS

            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < retries:
                        # Exponential backoff: delay * 2^attempt
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "function_retry",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=retries + 1,
                            error=str(e),
                            retry_in_seconds=round(wait_time, 2)
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            "function_failed_all_retries",
                            function=func.__name__,
                            attempts=retries + 1,
                            error=str(e)
                        )

            # If all retries failed, raise the last exception
            raise last_exception

        return wrapper
    return decorator
