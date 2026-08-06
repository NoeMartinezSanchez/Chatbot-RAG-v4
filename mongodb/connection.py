"""Conexión singleton a MongoDB Atlas usando motor (async)."""
import asyncio
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from config.settings import settings

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """Conexión singleton asíncrona a MongoDB.

    Patrón singleton: todas las llamadas a ``MongoDBConnection()``
    retornan la misma instancia y comparten el mismo cliente.
    """

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls) -> "MongoDBConnection":
        """Retorna la instancia única de la clase."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Inicializa los atributos solo la primera vez."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._client: Optional[AsyncIOMotorClient] = None
            self._db: Optional[AsyncIOMotorDatabase] = None
            self._indexes_ready = False

    async def connect(self) -> AsyncIOMotorDatabase:
        """Establece la conexión con MongoDB Atlas.

        Args:
            None.

        Returns:
            Instancia de la base de datos configurada (``MONGODB_DB_NAME``).

        Raises:
            PyMongoError: Si no se puede conectar o hacer ping al servidor.
        """
        client = self._client
        if client is None:
            client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                retryWrites=settings.MONGODB_RETRY_WRITES,
                appName="chatbot-rag-cluster",
            )
            self._client = client
            if settings.MONGODB_ENABLE_LOGGING:
                logger.info("✅ Cliente MongoDB creado (pool=%d, timeout=%dms)",
                            settings.MONGODB_MAX_POOL_SIZE, settings.MONGODB_TIMEOUT_MS)

        self._db = client[settings.MONGODB_DB_NAME]
        await self.ping()

        if not self._indexes_ready:
            try:
                await self._create_indexes()
                self._indexes_ready = True
            except Exception as e:
                logger.warning("⚠️ No se pudieron crear índices (la conexión sigue activa): %s", e)

        if settings.MONGODB_ENABLE_LOGGING:
            logger.info("✅ Conexión a MongoDB establecida: %s", settings.MONGODB_DB_NAME)
        return self._db

    async def ping(self) -> bool:
        """Hace ping al servidor de MongoDB.

        Returns:
            True si el servidor responde correctamente.

        Raises:
            RuntimeError: Si no hay cliente activo.
        """
        if self._client is None:
            raise RuntimeError("No hay conexión activa. Llama a connect() primero.")
        await self._client.admin.command("ping")
        return True

    def get_db(self) -> AsyncIOMotorDatabase:
        """Retorna la instancia de la base de datos.

        Returns:
            Instancia de la base de datos ya conectada.

        Raises:
            RuntimeError: Si la conexión aún no fue establecida.
        """
        if self._client is None or self._db is None:
            raise RuntimeError("No hay conexión activa. Llama a connect() primero.")
        return self._db

    async def disconnect(self) -> None:
        """Cierra la conexión con MongoDB."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            if settings.MONGODB_ENABLE_LOGGING:
                logger.info("🔌 Conexión a MongoDB cerrada")

    @property
    def is_connected(self) -> bool:
        """Indica si hay un cliente activo."""
        return self._client is not None

    async def _create_indexes(self) -> None:
        """Crea índices para todas las colecciones (según configuración)."""
        if self._db is None:
            raise RuntimeError("Base de datos no inicializada. Llama a connect() primero.")

        db = self._db

        async def _idx(collection: str, keys: list, unique: bool = False) -> None:
            await db[collection].create_index(keys, unique=unique)

        await asyncio.gather(
            _idx(settings.MONGODB_COLL_CONVERSATIONS, [("conversation_id", ASCENDING)], unique=True),
            _idx(settings.MONGODB_COLL_CONVERSATIONS, [("session_id", ASCENDING)]),
            _idx(settings.MONGODB_COLL_CONVERSATIONS, [("user_id", ASCENDING)]),
            _idx(settings.MONGODB_COLL_METRICS, [("request_timestamp", DESCENDING)]),
            _idx(settings.MONGODB_COLL_METRICS, [("session_id", ASCENDING)]),
            _idx(settings.MONGODB_COLL_METRICS, [("endpoint", ASCENDING)]),
            _idx(settings.MONGODB_COLL_FEEDBACK, [("conversation_id", ASCENDING)]),
            _idx(settings.MONGODB_COLL_FEEDBACK, [("session_id", ASCENDING)]),
            _idx(settings.MONGODB_COLL_USERS, [("user_id", ASCENDING)], unique=True),
            _idx(settings.MONGODB_COLL_SESSIONS, [("session_id", ASCENDING)], unique=True),
            _idx(settings.MONGODB_COLL_RAG_CACHE, [("cache_key", ASCENDING)], unique=True),
            _idx(settings.MONGODB_COLL_LOGS, [("timestamp", DESCENDING)]),
        )
        if settings.MONGODB_ENABLE_LOGGING:
            logger.info("🔍 Índices creados/verificados para las colecciones de MongoDB")
