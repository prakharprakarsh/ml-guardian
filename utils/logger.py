"""
ML Guardian — Structured Logging
Provides JSON-formatted logging for audit compliance.
"""
import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "agent": getattr(record, "agent", "system"),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def get_logger(name: str, agent: str = "system") -> logging.Logger:
    """Get a configured logger for an agent or module."""
    logger = logging.getLogger(f"ml_guardian.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Add agent context
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.agent = agent
        return record
    
    logging.setLogRecordFactory(record_factory)
    return logger
