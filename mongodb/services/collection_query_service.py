"""Servicio de consultas de solo lectura sobre las colecciones MongoDB.

Permite a la pestaña "Consulta" del dashboard ejecutar operaciones
``count`` / ``find`` / ``distinct`` sobre un conjunto permitido de
colecciones, con filtro/sort/limit opcionales. Es estrictamente de
solo lectura: nunca ejecuta insert/update/delete ni agregaciones.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from mongodb.connection import MongoDBConnection
from config.settings import settings

logger = logging.getLogger(__name__)

# Límites para evitar consultas demasiado pesadas
MAX_LIMIT = 500
DEFAULT_LIMIT = 50

# Operadores de MongoDB que permiten ejecución de JS arbitraria o
# cómputos peligrosos dentro de un find; se bloquean por seguridad.
BLOCKED_OPERATORS = ("$where", "$function", "$accumulator", "$expr", "$jsonSchema")


class CollectionQueryService:
    """Consultas de solo lectura sobre colecciones MongoDB."""

    @staticmethod
    def _allowed_collections() -> List[str]:
        """Retorna la lista de colecciones consultables (whitelist)."""
        return [
            settings.MONGODB_COLL_CONVERSATIONS,
            settings.MONGODB_COLL_METRICS,
            settings.MONGODB_COLL_FEEDBACK,
            settings.MONGODB_COLL_USERS,
            settings.MONGODB_COLL_SESSIONS,
            settings.MONGODB_COLL_RAG_CACHE,
            settings.MONGODB_COLL_LOGS,
        ]

    @staticmethod
    def _validate_collection(name: str) -> str:
        """Valida que la colección esté en la whitelist configurada.

        Args:
            name: Nombre de la colección.

        Returns:
            El nombre canónico de la colección.

        Raises:
            ValueError: Si la colección no está permitida.
        """
        allowed = CollectionQueryService._allowed_collections()
        if name not in allowed:
            raise ValueError(f"Colección '{name}' no permitida. Permitidas: {', '.join(allowed)}")
        return name

    @staticmethod
    def _validate_filter(filter_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Valida un documento de filtro, bloqueando operadores peligrosos.

        Args:
            filter_doc: Filtro MongoDB (puede incluir operadores $gte, $lt, etc.).

        Returns:
            El mismo filtro si es válido.

        Raises:
            ValueError: Si contiene operadores bloqueados o no es un dict.
        """
        if not isinstance(filter_doc, dict):
            raise ValueError("El filtro debe ser un objeto JSON (dict)")
        for op in BLOCKED_OPERATORS:
            if op in filter_doc:
                raise ValueError(f"Operador '{op}' no permitido en consultas")
            # Revisar recursivamente valores anidados
            for value in filter_doc.values():
                if isinstance(value, dict) and op in value:
                    raise ValueError(f"Operador '{op}' no permitido en consultas")
        return filter_doc

    @staticmethod
    def _jsonable(doc: Any) -> Any:
        """Convierte un documento BSON a tipos serializables JSON.

        Maneja ``ObjectId``, ``datetime`` y estructuras anidadas
        (dict/list). Los datetimes se convierten a string ISO localizado
        en la zona configurada (CDMX) usando ``format_local``.

        Args:
            doc: Valor a convertir (puede ser un documento completo).

        Returns:
            Valor JSON-serializable.
        """
        if hasattr(doc, "isoformat"):
            # datetime (naive UTC implícito o aware) -> ISO local
            try:
                from utils.timezones import format_local
                return format_local(doc, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return doc.isoformat()
        if isinstance(doc, dict):
            return {str(k): CollectionQueryService._jsonable(v) for k, v in doc.items()}
        if isinstance(doc, (list, tuple)):
            return [CollectionQueryService._jsonable(v) for v in doc]
        # ObjectId y otros tipos BSON: intentar str()
        return str(doc) if not isinstance(doc, (str, int, float, bool)) or doc is None else doc

    async def list_collections(self) -> List[Dict[str, Any]]:
        """Lista las colecciones permitidas con su conteo estimado.

        Returns:
            Lista de dicts ``{"name": ..., "label": ..., "doc_count": ...}``.
        """
        db = await MongoDBConnection().connect()
        result = []
        for name in self._allowed_collections():
            try:
                count = await db[name].estimated_document_count()
            except Exception as e:
                logger.debug("No se pudo contar la colección %s: %s", name, e)
                count = None
            result.append({
                "name": name,
                "label": name.replace("_", " ").title(),
                "doc_count": count,
            })
        return result

    async def run_query(
        self,
        collection: str,
        operation: str = "find",
        filter_doc: Optional[Dict[str, Any]] = None,
        sort_doc: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        distinct_field: Optional[str] = None,
        date_field: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ejecuta una consulta de solo lectura sobre una colección.

        Args:
            collection: Nombre de la colección (whitelist).
            operation: "count", "find" o "distinct".
            filter_doc: Filtro MongoDB opcional.
            sort_doc: Ordenamiento opcional (ej: {"created_at": -1}).
            limit: Máximo de documentos a devolver (tope MAX_LIMIT).
            distinct_field: Campo usado por la operación "distinct".
            date_field: Campo de fecha para filtrar "por fecha".
            date_from: ISO desde (incluyente) para date_field.
            date_to: ISO hasta (incluyente) para date_field.

        Returns:
            Dict con ``operation``, ``total`` (o count), ``docs``,
            ``fields``, ``took_ms`` y ``timestamp``.

        Raises:
            ValueError: Si la operación/salida no es válida.
        """
        name = self._validate_collection(collection)
        filter_doc = self._validate_filter(filter_doc or {})

        if operation not in ("count", "find", "distinct"):
            raise ValueError(f"Operación '{operation}' no soportada (count, find, distinct)")

        limit = min(int(limit) if limit else DEFAULT_LIMIT, MAX_LIMIT)
        if limit <= 0:
            limit = DEFAULT_LIMIT

        if operation == "distinct" and not distinct_field:
            raise ValueError("La operación 'distinct' requiere 'distinct_field'")

        # Filtro por fecha (rango) convertido a datetime real para que
        # coincida con los BSON Date almacenados en la base.
        if date_field:
            if not date_from and not date_to:
                raise ValueError("Para filtrar por fecha proporciona 'date_from' y/o 'date_to'")
            from datetime import datetime
            range_filter: Dict[str, Any] = {}
            if date_from:
                range_filter["$gte"] = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            if date_to:
                range_filter["$lte"] = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            if isinstance(filter_doc.get(date_field), dict):
                filter_doc[date_field] = {**filter_doc[date_field], **range_filter}
            else:
                filter_doc[date_field] = range_filter

        db = await MongoDBConnection().connect()
        col = db[name]
        start = time.time()

        total = 0
        docs: List[Dict[str, Any]] = []
        fields: List[str] = []

        if operation == "count":
            total = await col.count_documents(filter_doc)
        elif operation == "distinct":
            values = await col.distinct(distinct_field, filter_doc)
            docs = [{"valor": self._jsonable(v)} for v in values]
            total = len(docs)
            fields = ["valor"]
        else:  # find
            cursor = col.find(filter_doc)
            if sort_doc:
                _sort = [(str(k), 1 if v in (1, "asc", "ascending") else -1)
                         for k, v in (sort_doc or {}).items()]
                cursor = cursor.sort(_sort)
            total = await col.count_documents(filter_doc)
            raw = await cursor.limit(limit).to_list(length=None)
            docs = [self._jsonable(d) for d in raw]
            # Columnas: keys del primer documento (o del más reciente)
            if raw:
                fields = list(raw[0].keys())

        took_ms = round((time.time() - start) * 1000, 2)
        from utils.timezones import now_local
        return {
            "operation": operation,
            "collection": name,
            "total": total,
            "returned": len(docs),
            "fields": fields,
            "docs": docs,
            "took_ms": took_ms,
            "timestamp": now_local().isoformat(),
        }


# Instancia global para reutilizar en endpoints y scripts (convención)
collection_query_service = CollectionQueryService()