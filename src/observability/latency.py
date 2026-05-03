"""
Timing decorator for measuring pipeline stage latency.
"""
import time
import functools
import inspect

from src.observability.logger import logger


def timed(stage_name: str):
    """
    Decorator to measure and log execution time of a function.

    Supports both:
    - synchronous functions
    - asynchronous functions (FastAPI-ready)

    Logs:
    - stage name
    - latency (ms)
    - success / failure
    """

    def decorator(func):

        # Async function wrapper
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    logger.info(
                        "stage_completed",
                        stage=stage_name,
                        latency_ms=round(elapsed_ms, 1),
                        success=True,
                    )

                    return result

                except Exception:
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    logger.exception(
                        "stage_failed",
                        stage=stage_name,
                        latency_ms=round(elapsed_ms, 1),
                        success=False,
                    )
                    raise

            return async_wrapper

        # Sync function wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                elapsed_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    "stage_completed",
                    stage=stage_name,
                    latency_ms=round(elapsed_ms, 1),
                    success=True,
                )

                return result

            except Exception:
                elapsed_ms = (time.perf_counter() - start) * 1000

                logger.exception(
                    "stage_failed",
                    stage=stage_name,
                    latency_ms=round(elapsed_ms, 1),
                    success=False,
                )
                raise

        return sync_wrapper

    return decorator