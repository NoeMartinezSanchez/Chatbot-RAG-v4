"""Repositorio base genérico para MongoDB."""
import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """Repositorio genérico de operaciones CRUD para una colección MongoDB.

    Attributes:
        collection: Colección MongoDB subyacente.
        model: Modelo Pydantic usado para serializar/deserializar documentos.
    """

    def __init__(self, collection: AsyncIOMotorCollection, model: Type[T]) -> None:
        """Inicializa el repositorio.

        Args:
            collection: Colección MongoDB a operar.
            model: Clase del modelo Pydantic de los documentos.
        """
        self._collection = collection
        self._model = model

    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Retorna la colección MongoDB subyacente."""
        return self._collection

    async def create(self, document: T) -> T:
        """Inserta un documento en la colección.

        Args:
            document: Instancia del modelo a insertar.

        Returns:
            El documento insertado, incluyendo su ``id`` de MongoDB.
        """
        data = document.model_dump()
        result = await self._collection.insert_one(data)
        logger.info("📝 Documento creado: %s", result.inserted_id)
        return self._to_model({**data, "id": str(result.inserted_id)})

    async def create_many(self, documents: List[T]) -> List[T]:
        """Inserta múltiples documentos de una sola vez.

        Args:
            documents: Lista de instancias del modelo a insertar.

        Returns:
            Lista de documentos insertados, incluyendo sus ``id``.
        """
        if not documents:
            return []
        data_list = [d.model_dump() for d in documents]
        result = await self._collection.insert_many(data_list)
        stored = []
        for data, inserted_id in zip(data_list, result.inserted_ids):
            stored.append(self._to_model({**data, "id": str(inserted_id)}))
        logger.info("📝 %d documentos creados", len(stored))
        return stored

    async def find_one(self, filter: Dict[str, Any]) -> Optional[T]:
        """Busca un documento que coincida con el filtro.

        Args:
            filter: Filtro MongoDB (diccionario de criterios).

        Returns:
            El primer documento encontrado o ``None`` si no hay coincidencias.
        """
        doc = await self._collection.find_one(filter)
        if doc is None:
            return None
        return self._to_model(doc)

    async def find_many(
        self,
        filter: Dict[str, Any],
        limit: int = 50,
        skip: int = 0,
        sort: Optional[List[tuple]] = None,
    ) -> List[T]:
        """Busca múltiples documentos con paginación y ordenamiento.

        Args:
            filter: Filtro MongoDB (diccionario de criterios).
            limit: Número máximo de resultados (default: 50).
            skip: Documentos a omitir para paginación.
            sort: Lista de tuplas (campo, dirección) para ordenar.

        Returns:
            Lista de documentos encontrados.
        """
        cursor = self._collection.find(filter).skip(skip).limit(limit)
        if sort:
            cursor = cursor.sort(sort)
        docs = await cursor.to_list(length=limit)
        return [self._to_model(doc) for doc in docs]

    async def update_one(
        self,
        filter: Dict[str, Any],
        data: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """Actualiza un documento que coincida con el filtro.

        Args:
            filter: Filtro MongoDB para localizar el documento.
            data: Campos a actualizar (se aplican con ``$set``).
            upsert: Si True, inserta el documento si no existe (default: False).

        Returns:
            True si algún documento fue modificado.
        """
        result = await self._collection.update_one(filter, {"$set": data}, upsert=upsert)
        return result.modified_count > 0

    async def delete_one(self, filter: Dict[str, Any]) -> bool:
        """Elimina un documento que coincida con el filtro.

        Args:
            filter: Filtro MongoDB para localizar el documento.

        Returns:
            True si algún documento fue eliminado.
        """
        result = await self._collection.delete_one(filter)
        return result.deleted_count > 0

    async def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Cuenta documentos que coinciden con el filtro.

        Args:
            filter: Filtro MongoDB (opcional; vacío cuenta todo).

        Returns:
            Número de documentos que coinciden.
        """
        return await self._collection.count_documents(filter or {})

    def _to_model(self, doc: Dict[str, Any]) -> T:
        """Convierte un documento MongoDB (con ``_id``) al modelo Pydantic.

        Args:
            doc: Documento crudo proveniente de MongoDB.

        Returns:
            Instancia del modelo Pydantic con ``id`` poblado.
        """
        data = dict(doc)
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return self._model.model_validate(data)
