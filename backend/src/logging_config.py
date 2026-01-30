"""
Centralized Logging Configuration for Lung Cancer Assistant
============================================================

This module provides:
1. Structured JSON logging for production
2. Colored console logging for development
3. Rotating file handlers for persistent logs
4. LangSmith/LangChain tracing integration
5. Performance tracking decorators

Usage:
    from backend.src.logging_config import setup_logging, get_logger

    # Initialize logging (call once at startup)
    setup_logging()

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Processing patient", extra={"patient_id": "LC-001"})
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import time
import traceback

# ==================== Configuration ====================

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file paths
LOG_FILE_APP = LOGS_DIR / "app.log"
LOG_FILE_ERROR = LOGS_DIR / "error.log"
LOG_FILE_AGENT = LOGS_DIR / "agents.log"
LOG_FILE_API = LOGS_DIR / "api.log"
LOG_FILE_WORKFLOW = LOGS_DIR / "workflow.log"
LOG_FILE_DEBUG = LOGS_DIR / "debug.log"


# ==================== Custom Formatters ====================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "patient_id"):
            log_data["patient_id"] = record.patient_id
        if hasattr(record, "agent_name"):
            log_data["agent_name"] = record.agent_name
        if hasattr(record, "workflow_id"):
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Build message parts
        parts = [
            f"{color}{self.BOLD}[{record.levelname:8}]{self.RESET}",
            f"\033[90m{timestamp}\033[0m",
            f"\033[94m{record.name}\033[0m",
            f"{record.getMessage()}"
        ]

        # Add extra context if available
        extra_parts = []
        if hasattr(record, "patient_id"):
            extra_parts.append(f"patient={record.patient_id}")
        if hasattr(record, "agent_name"):
            extra_parts.append(f"agent={record.agent_name}")
        if hasattr(record, "duration_ms"):
            extra_parts.append(f"duration={record.duration_ms}ms")

        if extra_parts:
            parts.append(f"\033[90m[{', '.join(extra_parts)}]\033[0m")

        message = " | ".join(parts)

        # Add exception if present
        if record.exc_info:
            message += f"\n{color}{self.formatException(record.exc_info)}{self.RESET}"

        return message


# ==================== LangSmith/LangChain Tracing ====================

def setup_langsmith_tracing():
    """
    Configure LangSmith tracing for LangChain/LangGraph observability.

    Set these environment variables:
    - LANGCHAIN_TRACING_V2=true
    - LANGCHAIN_API_KEY=your_api_key
    - LANGCHAIN_PROJECT=LungCancerAssistant
    - LANGCHAIN_ENDPOINT=https://api.smith.langchain.com (optional)
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if tracing_enabled and api_key:
        # Set required environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LungCancerAssistant")

        # Optional: Set endpoint (defaults to LangSmith cloud)
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

        # Enable LangChain debug mode if requested
        if os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true":
            try:
                from langchain.globals import set_debug, set_verbose
                set_debug(True)
                set_verbose(True)
            except ImportError:
                pass

        logging.getLogger("lca.tracing").info(
            f"LangSmith tracing enabled - Project: {os.environ['LANGCHAIN_PROJECT']}"
        )
        return True
    else:
        logging.getLogger("lca.tracing").info(
            "LangSmith tracing disabled - Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable"
        )
        return False


# ==================== Logger Setup ====================

def setup_logging(
    log_level: Optional[str] = None,
    enable_json: bool = False,
    enable_file_logging: bool = True,
    enable_langsmith: bool = True
) -> None:
    """
    Initialize centralized logging configuration.

    Args:
        log_level: Override log level (default from LOG_LEVEL env var or INFO)
        enable_json: Use JSON format for file logs (recommended for production)
        enable_file_logging: Enable file-based logging
        enable_langsmith: Enable LangSmith tracing
    """
    # Determine log level
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # File handlers (if enabled)
    if enable_file_logging:
        # Main application log (rotating, 10MB per file, keep 10 files)
        app_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_APP,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        app_handler.setLevel(level)
        app_handler.setFormatter(
            JSONFormatter() if enable_json else
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(app_handler)

        # Error log (errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE_ERROR,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)

        # Debug log (all levels, larger rotation)
        if level <= logging.DEBUG:
            debug_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE_DEBUG,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(debug_handler)

    # Configure specific loggers
    _configure_component_loggers(level, enable_json, enable_file_logging)

    # Suppress noisy third-party loggers
    _suppress_noisy_loggers()

    # Setup LangSmith tracing
    if enable_langsmith:
        setup_langsmith_tracing()

    # Log startup
    logging.getLogger("lca.startup").info(
        f"Logging initialized - Level: {level_str}, JSON: {enable_json}, "
        f"Files: {enable_file_logging}, Logs dir: {LOGS_DIR}"
    )


def _configure_component_loggers(level: int, enable_json: bool, enable_file_logging: bool) -> None:
    """Configure component-specific loggers with dedicated file handlers"""

    if not enable_file_logging:
        return

    formatter = JSONFormatter() if enable_json else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Agent logger
    agent_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_AGENT,
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    agent_handler.setLevel(level)
    agent_handler.setFormatter(formatter)

    for agent_logger_name in ['lca.agents', 'backend.src.agents']:
        agent_logger = logging.getLogger(agent_logger_name)
        agent_logger.addHandler(agent_handler)

    # API logger
    api_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_API,
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    api_handler.setLevel(level)
    api_handler.setFormatter(formatter)

    for api_logger_name in ['lca.api', 'backend.src.api']:
        api_logger = logging.getLogger(api_logger_name)
        api_logger.addHandler(api_handler)

    # Workflow logger
    workflow_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE_WORKFLOW,
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    workflow_handler.setLevel(level)
    workflow_handler.setFormatter(formatter)

    for workflow_logger_name in ['lca.workflow', 'backend.src.agents.lca_workflow']:
        workflow_logger = logging.getLogger(workflow_logger_name)
        workflow_logger.addHandler(workflow_handler)


def _suppress_noisy_loggers() -> None:
    """Suppress verbose third-party loggers"""
    noisy_loggers = [
        'neo4j',
        'neo4j.pool',
        'neo4j.io',
        'sentence_transformers',
        'httpx',
        'httpcore',
        'urllib3',
        'asyncio',
        'uvicorn.access',
        'uvicorn.error',
        'watchfiles',
        'filelock',
        'huggingface_hub',
        'transformers',
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Neo4j is especially noisy
    logging.getLogger('neo4j').setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the standardized naming convention.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"patient_id": "LC-001"})
    """
    # Standardize logger names
    if name.startswith('backend.src.'):
        # Convert to lca.* format
        standardized = name.replace('backend.src.', 'lca.')
    elif not name.startswith('lca.'):
        standardized = f"lca.{name}"
    else:
        standardized = name

    return logging.getLogger(standardized)


# ==================== Decorators ====================

def log_execution(logger_name: Optional[str] = None):
    """
    Decorator to log function execution with timing.

    Example:
        @log_execution()
        def process_patient(patient_id: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            func_name = func.__name__

            logger.debug(f"Entering {func_name}", extra={
                "function": func_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(f"Completed {func_name}", extra={
                    "function": func_name,
                    "duration_ms": round(duration_ms, 2),
                    "success": True
                })
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Failed {func_name}: {str(e)}", extra={
                    "function": func_name,
                    "duration_ms": round(duration_ms, 2),
                    "success": False,
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            func_name = func.__name__

            logger.debug(f"Entering {func_name}", extra={
                "function": func_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(f"Completed {func_name}", extra={
                    "function": func_name,
                    "duration_ms": round(duration_ms, 2),
                    "success": True
                })
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Failed {func_name}: {str(e)}", extra={
                    "function": func_name,
                    "duration_ms": round(duration_ms, 2),
                    "success": False,
                    "error_type": type(e).__name__
                }, exc_info=True)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def log_agent_action(agent_name: str):
    """
    Decorator for logging agent actions.

    Example:
        @log_agent_action("ClassificationAgent")
        def classify_patient(self, state):
            ...
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"lca.agents.{agent_name.lower()}")

            logger.info(f"Agent action started: {func.__name__}", extra={
                "agent_name": agent_name,
                "action": func.__name__
            })

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(f"Agent action completed: {func.__name__}", extra={
                    "agent_name": agent_name,
                    "action": func.__name__,
                    "duration_ms": round(duration_ms, 2)
                })
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Agent action failed: {func.__name__} - {str(e)}", extra={
                    "agent_name": agent_name,
                    "action": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e)
                }, exc_info=True)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(f"lca.agents.{agent_name.lower()}")

            logger.info(f"Agent action started: {func.__name__}", extra={
                "agent_name": agent_name,
                "action": func.__name__
            })

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(f"Agent action completed: {func.__name__}", extra={
                    "agent_name": agent_name,
                    "action": func.__name__,
                    "duration_ms": round(duration_ms, 2)
                })
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Agent action failed: {func.__name__} - {str(e)}", extra={
                    "agent_name": agent_name,
                    "action": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e)
                }, exc_info=True)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ==================== Context Managers ====================

class LogContext:
    """
    Context manager for adding context to log messages.

    Example:
        with LogContext(patient_id="LC-001", workflow_id="WF-123"):
            logger.info("Processing patient")  # Will include patient_id and workflow_id
    """

    _context: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        self.new_context = kwargs
        self.old_context = {}

    def __enter__(self):
        self.old_context = LogContext._context.copy()
        LogContext._context.update(self.new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        LogContext._context = self.old_context
        return False

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        return cls._context.copy()


# ==================== Utility Functions ====================

def log_patient_event(
    logger: logging.Logger,
    event: str,
    patient_id: str,
    level: int = logging.INFO,
    **extra
) -> None:
    """
    Log a patient-related event with standardized format.

    Args:
        logger: Logger instance
        event: Event description
        patient_id: Patient identifier
        level: Log level
        **extra: Additional context
    """
    logger.log(level, f"[Patient:{patient_id}] {event}", extra={
        "patient_id": patient_id,
        **extra
    })


def log_workflow_event(
    logger: logging.Logger,
    event: str,
    workflow_id: str,
    agent_name: Optional[str] = None,
    level: int = logging.INFO,
    **extra
) -> None:
    """
    Log a workflow-related event with standardized format.

    Args:
        logger: Logger instance
        event: Event description
        workflow_id: Workflow identifier
        agent_name: Current agent name (optional)
        level: Log level
        **extra: Additional context
    """
    context = {"workflow_id": workflow_id, **extra}
    if agent_name:
        context["agent_name"] = agent_name

    logger.log(level, f"[Workflow:{workflow_id}] {event}", extra=context)


# ==================== SSE Log Streaming ====================

class SSELogCaptureHandler(logging.Handler):
    """
    Custom logging handler that captures logs for SSE streaming to frontend.

    Usage:
        async def my_stream():
            handler = SSELogCaptureHandler(level=logging.INFO)
            handler.install()  # Start capturing
            try:
                # Do work that generates logs
                async for log_entry in handler.stream_logs():
                    yield format_sse({"type": "log", "content": log_entry})
            finally:
                handler.uninstall()  # Stop capturing
    """

    def __init__(self, level=logging.INFO, filter_prefixes=None):
        super().__init__(level)
        self.log_queue = None  # Will be set to asyncio.Queue when installed
        self.filter_prefixes = filter_prefixes or [
            'lca.agents', 'lca.workflow', 'lca.services',
            'backend.src.agents', 'backend.src.services'
        ]
        self._installed = False
        self._original_handlers = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Capture log record to queue for streaming"""
        if self.log_queue is None:
            return

        # Filter by logger name prefix
        should_capture = any(
            record.name.startswith(prefix)
            for prefix in self.filter_prefixes
        )

        if not should_capture:
            return

        try:
            # Format the log entry for frontend display
            log_entry = {
                "timestamp": datetime.now().strftime('%H:%M:%S.%f')[:-3],
                "level": record.levelname,
                "logger": record.name.replace('lca.', '').replace('backend.src.', ''),
                "message": record.getMessage(),
            }

            # Add extra context if available
            if hasattr(record, 'agent_name'):
                log_entry["agent"] = record.agent_name
            if hasattr(record, 'patient_id'):
                log_entry["patient_id"] = record.patient_id
            if hasattr(record, 'duration_ms'):
                log_entry["duration_ms"] = record.duration_ms

            # Put in queue (non-blocking)
            try:
                self.log_queue.put_nowait(log_entry)
            except:
                pass  # Queue full, skip this log

        except Exception:
            pass  # Don't let logging errors break the app

    def install(self, queue) -> None:
        """Install handler on relevant loggers"""
        import asyncio
        self.log_queue = queue

        # Add handler to root logger and specific loggers
        loggers_to_capture = [
            logging.getLogger(),  # Root logger
            logging.getLogger('lca'),
            logging.getLogger('lca.agents'),
            logging.getLogger('lca.workflow'),
            logging.getLogger('lca.services'),
            logging.getLogger('backend.src.agents'),
            logging.getLogger('backend.src.services'),
        ]

        for logger in loggers_to_capture:
            if self not in logger.handlers:
                logger.addHandler(self)

        self._installed = True

    def uninstall(self) -> None:
        """Remove handler from loggers"""
        if not self._installed:
            return

        loggers_to_capture = [
            logging.getLogger(),
            logging.getLogger('lca'),
            logging.getLogger('lca.agents'),
            logging.getLogger('lca.workflow'),
            logging.getLogger('lca.services'),
            logging.getLogger('backend.src.agents'),
            logging.getLogger('backend.src.services'),
        ]

        for logger in loggers_to_capture:
            if self in logger.handlers:
                logger.removeHandler(self)

        self.log_queue = None
        self._installed = False

    async def stream_logs(self, timeout: float = 0.1):
        """
        Async generator that yields log entries from the queue.

        Args:
            timeout: How long to wait for each log entry

        Yields:
            dict: Log entry dictionaries
        """
        import asyncio

        if self.log_queue is None:
            return

        while True:
            try:
                log_entry = await asyncio.wait_for(
                    self.log_queue.get(),
                    timeout=timeout
                )
                yield log_entry
            except asyncio.TimeoutError:
                # No log available, yield control
                break
            except Exception:
                break


def create_sse_log_handler(level=logging.INFO) -> SSELogCaptureHandler:
    """Factory function to create SSE log capture handler"""
    return SSELogCaptureHandler(level=level)


# ==================== Auto-initialization ====================

# Auto-setup logging when module is imported (can be overridden)
if os.getenv("LCA_AUTO_LOGGING", "true").lower() == "true":
    # Only auto-setup if not already configured
    if not logging.getLogger().handlers:
        setup_logging()
