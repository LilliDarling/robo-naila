"""Centralized logging utility for AI Server using structlog

Provides structured logging with better context management and cleaner output.
"""

import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    colored: bool = True,
    json_logs: bool = False
):
    """Setup centralized structlog configuration

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        colored: Enable colored console output
        json_logs: Output JSON format instead of console format
    """
    # Processors for all scenarios
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # Console or JSON rendering
    if json_logs:
        # JSON output for production/file logging
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console output with colors
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=colored)
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_factory = structlog.PrintLoggerFactory(file=open(log_file, "a"))
        structlog.configure(logger_factory=file_factory)


def get_logger(name: str):
    """Get a structlog logger with the specified name

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        Configured structlog logger instance
    """
    return structlog.get_logger(name)


def set_module_level(module_name: str, level: str):
    """Set logging level for a specific module

    Args:
        module_name: Module name (e.g., 'services.tts')
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = structlog.get_logger(module_name)
    logger = logger.bind(log_level=level)


def silence_module(module_name: str):
    """Silence all logging from a specific module

    Args:
        module_name: Module name to silence
    """
    set_module_level(module_name, "CRITICAL")


def silence_noisy_modules():
    """Silence commonly noisy third-party modules"""
    noisy_modules = [
        "urllib3",
        "httpx",
        "httpcore",
        "asyncio",
        "PIL",
        "matplotlib",
        "paho",
    ]

    for module in noisy_modules:
        silence_module(module)


# Convenience functions for common logging patterns
def log_startup(logger, component: str, **details):
    """Log component startup with structured context

    Args:
        logger: Structlog logger instance
        component: Component name
        **details: Additional context as keyword arguments
    """
    logger.info("starting", component=component, **details)


def log_shutdown(logger, component: str):
    """Log component shutdown

    Args:
        logger: Structlog logger instance
        component: Component name
    """
    logger.info("stopping", component=component)


def log_performance(logger, operation: str, duration_ms: int, **details):
    """Log performance metrics with structured context

    Args:
        logger: Structlog logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **details: Additional context as keyword arguments
    """
    logger.debug("performance", operation=operation, duration_ms=duration_ms, **details)


def log_error_with_context(logger, error: Exception, context: str = "", **extra):
    """Log error with structured context

    Args:
        logger: Structlog logger instance
        error: Exception object
        context: Context description
        **extra: Additional context as keyword arguments
    """
    logger.error(
        "error",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context,
        **extra
    )


def log_model_load(logger, model_name: str, duration_s: float, success: bool = True):
    """Log model loading with structured context

    Args:
        logger: Structlog logger instance
        model_name: Model name/path
        duration_s: Load duration in seconds
        success: Whether loading succeeded
    """
    if success:
        logger.info("model_loaded", model=model_name, duration_s=duration_s)
    else:
        logger.error("model_load_failed", model=model_name)


def log_config(logger, config: dict):
    """Log configuration settings with structured context

    Args:
        logger: Structlog logger instance
        config: Configuration dictionary
    """
    logger.info("configuration", **config)
