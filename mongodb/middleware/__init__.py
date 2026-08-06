"""Middleware para la capa MongoDB."""
from mongodb.middleware.logging_middleware import LoggingMiddleware

__all__ = ["LoggingMiddleware"]
