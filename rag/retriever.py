"""
Vector Store usando FAISS en lugar de ChromaDB
"""
import faiss
import numpy as np
import pickle
import json
import os
import re
import unicodedata
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from config.settings import settings  # <-- SE AÑADIO ESTA LINEA

logger = logging.getLogger(__name__)

# --- Placeholder resolution cache ---
_RESOLVER_URL_MAP: Optional[Dict[str, str]] = None
_RESOLVER_FECHA_MAP: Optional[Dict[str, Dict[str, str]]] = None
_RESOLVER_PATRON_MAP: Optional[Dict[str, str]] = None
_RESOLVER_PATRON_COMPILED: Optional[list] = None

# Marcadores residuales que NUNCA deben llegar al usuario (X-run tipo XX / XXXX)
_PATRON_X_RESIDUAL = re.compile(r"\bX{2,}\b")

# Clases de acentos para construir regex insensibles a acentos (español)
_ACCENT_CLASSES = {
    "a": "[aá]", "e": "[eé]", "i": "[ií]", "o": "[oó]", "u": "[uúü]", "n": "[nñ]",
}


def _base_letter(ch: str) -> str:
    """Devuelve la letra base sin acento (ó -> o, ñ -> n)."""
    descompuesto = unicodedata.normalize("NFD", ch)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _patron_a_regex(patron: str):
    """Construye una regex insensible a acentos para un patrón con comodines.

    Convierte ``Generación XX`` en ``Generaci[oó]n X{2,}`` para que coincida
    tanto con ``Generación XX`` como con ``Generacion XX``.
    """
    partes = []
    i = 0
    while i < len(patron):
        ch = patron[i]
        if ch in ("X", "x"):
            j = i
            while j < len(patron) and patron[j] in ("X", "x"):
                j += 1
            partes.append(r"X{2,}")
            i = j
            continue
        base = _base_letter(ch).lower()
        if base in _ACCENT_CLASSES:
            clase = _ACCENT_CLASSES[base]
            partes.append(clase if ch.islower() else clase.upper())
        else:
            partes.append(re.escape(ch))
        i += 1
    return re.compile("".join(partes), re.IGNORECASE)


def get_placeholder_maps() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Retorna los mapas actuales de placeholders (URLs y fechas), solo lectura.

    Devuelve copias de los mapas cacheados usados por ``resolver_placeholders``
    para permitir auditar la resolución (p. ej. persistir ``placeholders_resolved``)
    sin modificar el pipeline de retrieval.

    Returns:
        Tupla ``(mapa_urls, mapa_fechas)``: URL key → valor; fecha key → dict
        (con la clave ``valor_actual``).
    """
    _load_resolver_maps()
    return (dict(_RESOLVER_URL_MAP or {}), dict(_RESOLVER_FECHA_MAP or {}))

def _load_resolver_maps():
    global _RESOLVER_URL_MAP, _RESOLVER_FECHA_MAP, _RESOLVER_PATRON_MAP, _RESOLVER_PATRON_COMPILED
    if _RESOLVER_URL_MAP is None:
        url_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mapeo_urls_global.json')
        try:
            with open(url_path, 'r', encoding='utf-8') as f:
                _RESOLVER_URL_MAP = json.load(f)
            logger.info(f"Loaded URL map with {len(_RESOLVER_URL_MAP)} entries")
        except Exception as e:
            logger.error(f"Failed to load URL map: {e}")
            _RESOLVER_URL_MAP = {}
    if _RESOLVER_FECHA_MAP is None:
        fecha_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mapeo_fechas_completo.json')
        try:
            with open(fecha_path, 'r', encoding='utf-8') as f:
                _RESOLVER_FECHA_MAP = json.load(f)
            logger.info(f"Loaded fecha map with {len(_RESOLVER_FECHA_MAP)} entries")
        except Exception as e:
            logger.error(f"Failed to load fecha map: {e}")
            _RESOLVER_FECHA_MAP = {}
    if _RESOLVER_PATRON_MAP is None:
        # Mapea los patrones con comodines (p. ej. "del XX al XX de XXXX de XXXX")
        # al valor actual. Algunos chunks guardaron el PATRÓN en lugar de la clave
        # (fechaN), por lo que sin esto el usuario veía "XX" en la respuesta.
        patron_map: Dict[str, str] = {}
        base = os.path.dirname(os.path.dirname(__file__))
        for nombre in ('mapeo_fechas_completo.json', 'mapeo_fechas_categorizadas.json'):
            try:
                with open(os.path.join(base, 'data', nombre), 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue
            for entry in data.values():
                if not isinstance(entry, dict):
                    continue
                patron = entry.get('patron')
                valor = entry.get('valor_actual')
                if patron and valor:
                    patron_map[patron] = valor
        _RESOLVER_PATRON_MAP = patron_map
        _RESOLVER_PATRON_COMPILED = [
            (_patron_a_regex(patron), valor)
            for patron, valor in sorted(patron_map.items(), key=lambda kv: len(kv[0]), reverse=True)
        ]
        logger.info(f"Loaded patron map with {len(_RESOLVER_PATRON_MAP)} entries")

def resolver_placeholders(text: str) -> str:
    """Replace urlN and fechaN placeholders in text with actual values.

    Además de las claves (``fecha1``, ``url1``), sustituye los PATRONES con
    comodines (``del XX al XX de XXXX de XXXX``, ``Generación XX``) por su
    ``valor_actual``, porque algunos chunks almacenaron el patrón en lugar de
    la clave. Es idempotente y no rompe el pipeline si faltan los mapas.

    Args:
        text: Texto que puede contener placeholders o patrones sin resolver.

    Returns:
        Texto con los placeholders/patrones resueltos.
    """
    _load_resolver_maps()
    result = text

    # 0) Sustituir PATRONES con comodines (más largos primero para no solapar)
    if _RESOLVER_PATRON_COMPILED:
        for patron_re, valor in _RESOLVER_PATRON_COMPILED:
            result = patron_re.sub(valor, result)

    # 1) Replace URL placeholders (longest keys first to avoid partial overlap)
    if _RESOLVER_URL_MAP:
        for key in sorted(_RESOLVER_URL_MAP.keys(), key=len, reverse=True):
            result = result.replace(key, _RESOLVER_URL_MAP[key])

    # 2) Replace fecha placeholders (longest keys first to avoid partial overlap)
    if _RESOLVER_FECHA_MAP:
        for key in sorted(_RESOLVER_FECHA_MAP.keys(), key=len, reverse=True):
            entry = _RESOLVER_FECHA_MAP[key]
            valor = entry.get('valor_actual', '')
            if valor:
                result = result.replace(key, valor)

    if result != text:
        logger.info(f"Placeholders resolved in text (len before: {len(text)}, after: {len(result)})")

    return result


def limpiar_marcadores_residuales(text: str) -> str:
    """Elimina marcadores de plantilla que hayan quedado sin resolver.

    Red de seguridad: si tras :func:`resolver_placeholders` aún aparece un
    bloque de comodines (``XX``/``XXXX``) o una clave sin valor, se retira para
    que el usuario NUNCA vea ``Generación XX`` o ``del XX al XX``.

    Args:
        text: Respuesta final del chatbot.

    Returns:
        Texto limpio y con espacios normalizados.
    """
    if not text:
        return text
    result = text
    # Claves residuales (url1, fecha3, ...)
    result = re.sub(r"\b(?:url|fecha)\d+\b", "", result, flags=re.IGNORECASE)
    # Bloques de comodines: "del XX al XX de XXXX..." -> frase neutral
    result = re.sub(r"\bdel\s+X{2,}\s+al\s+X{2,}(?:\s+de\s+X{2,})*", "en fechas por confirmar", result, flags=re.IGNORECASE)
    result = re.sub(r"\bGeneraci[oó]n\s+X{2,}", "próxima generación", result, flags=re.IGNORECASE)
    # Cualquier X-run restante
    result = _PATRON_X_RESIDUAL.sub("", result)
    # Normalizar espacios y puntuación colgante
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\s+([,.;:])", r"\1", result)
    result = re.sub(r"([,.;:])\1+", r"\1", result)
    result = result.replace(" .", ".").replace(" ,", ",").strip()
    return result

class VectorStoreFAISS:
    """
    Almacén vectorial optimizado para CPU usando FAISS.
    Compatible con AWS Lambda + S3 para futura migración.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Inicializar almacén vectorial FAISS.
        
        Args:
            persist_directory: Directorio para persistir índices
        """
        if persist_directory is None:
            persist_directory = settings.FAISS_PERSIST_DIR
        
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Rutas de archivos
        self.index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.documents_path = os.path.join(persist_directory, "documents.pkl")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        self.intents_path = os.path.join(persist_directory, "intents.json")
        
        # Configuración
        self.embedding_dim = 384  # Dimensión de MiniLM
        
        # Datos en memoria
        self.index = None
        self.documents = []      # Lista de textos completos
        self.metadata = []       # Lista de metadatos
        self.intents = {}        # Datos de intents
        self.doc_id_to_idx = {}  # Mapeo ID → índice
        
        # Estadísticas
        self.stats = {
            "total_documents": 0,
            "last_updated": None
        }
        
        # Cargar datos existentes
        self._load_existing()
        logger.info(f"VectorStoreFAISS inicializado. Documentos: {len(self.documents)}")
    
    def _load_existing(self):
        """Cargar datos existentes desde disco"""
        try:
            # Cargar índice FAISS
            if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Índice FAISS cargado: {self.index.ntotal} vectores")
            
            # Cargar documentos
            if os.path.exists(self.documents_path) and os.path.getsize(self.documents_path) > 0:
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            # Cargar metadatos
            if os.path.exists(self.metadata_path) and os.path.getsize(self.metadata_path) > 0:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            # Cargar intents
            if os.path.exists(self.intents_path) and os.path.getsize(self.intents_path) > 0:
                with open(self.intents_path, 'r', encoding='utf-8') as f:
                    self.intents = json.load(f)
            
            # Reconstruir mapeo ID → índice
            for idx, meta in enumerate(self.metadata):
                if "doc_id" in meta:
                    self.doc_id_to_idx[meta["doc_id"]] = idx
            
            self.stats["total_documents"] = len(self.documents)
            self.stats["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos existentes: {e}")
            # Inicializar vacío
            self.index = None
            self.documents = []
            self.metadata = []
            self.intents = {"intents": []}
    
    def _save(self):
        """Guardar todos los datos a disco"""
        try:
            # Guardar índice FAISS
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
            
            # Guardar documentos
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Guardar metadatos
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Guardar intents
            with open(self.intents_path, 'w', encoding='utf-8') as f:
                json.dump(self.intents, f, ensure_ascii=False, indent=2)
            
            self.stats["last_updated"] = datetime.now().isoformat()
            logger.debug("Datos guardados en disco")
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
    
    def store_intents(self, intents_file: str):
        """
        Cargar y almacenar intents desde archivo JSON.
        
        Args:
            intents_file: Ruta al archivo intents.json
        """
        try:
            with open(intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            
            # Guardar copia local
            with open(self.intents_path, 'w', encoding='utf-8') as f:
                json.dump(self.intents, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cargados {len(self.intents.get('intents', []))} intents")
            
        except Exception as e:
            logger.error(f"Error cargando intents: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Añadir documentos con embeddings al índice.
        
        Args:
            documents: Lista de dicts con 'content' y 'metadata'
            embeddings: Array numpy de forma (n_docs, embedding_dim)
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Número de documentos ({len(documents)}) no coincide con embeddings ({embeddings.shape[0]})")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Dimensión de embeddings ({embeddings.shape[1]}) no coincide con {self.embedding_dim}")
        
        # Inicializar índice si no existe
        if self.index is None:
            # Índice FlatL2 para similitud coseno (normalizamos embeddings)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Añadir embeddings al índice
        self.index.add(embeddings.astype('float32'))
        
        # Almacenar documentos y metadatos
        for i, doc in enumerate(documents):
            doc_id = hashlib.md5(doc['content'].encode()).hexdigest()[:12]
            
            # Guardar documento
            self.documents.append(doc['content'])
            
            # Guardar metadata con ID único
            metadata = doc.get('metadata', {})
            metadata.update({
                "doc_id": doc_id,
                "added_at": datetime.now().isoformat(),
                "doc_index": len(self.documents) - 1
            })
            self.metadata.append(metadata)
            
            # Actualizar mapeo
            self.doc_id_to_idx[doc_id] = len(self.documents) - 1
        
        # Actualizar estadísticas
        self.stats["total_documents"] = len(self.documents)
        
        # Guardar
        self._save()
        logger.info(f"Añadidos {len(documents)} documentos. Total: {len(self.documents)}")
    
    def add_document(self, content: str, metadata: Optional[Dict] = None, embedding: Optional[np.ndarray] = None):
        """
        Añadir un solo documento.
        
        Args:
            content: Texto del documento
            metadata: Metadatos opcionales
            embedding: Embedding pre-calculado (opcional)
        """
        if metadata is None:
            metadata = {}
        
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Si tenemos embedding, añadirlo al índice
        if embedding is not None:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding debe tener dimensión {self.embedding_dim}")
            
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        # Almacenar documento
        self.documents.append(content)
        
        # Guardar metadata
        metadata.update({
            "doc_id": doc_id,
            "added_at": datetime.now().isoformat(),
            "doc_index": len(self.documents) - 1
        })
        self.metadata.append(metadata)
        self.doc_id_to_idx[doc_id] = len(self.documents) - 1
        
        # Actualizar estadísticas
        self.stats["total_documents"] = len(self.documents)
        
        # Guardar
        self._save()
        logger.info(f"Documento añadido: {metadata.get('title', 'Sin título')} (ID: {doc_id})")
    
    def search_intents(self, query_text: str = None, query_embedding: np.ndarray = None, top_k: int = 1) -> Dict:
        """
        Buscar intents similares usando matching por texto.
        
        Args:
            query_text: Texto de la consulta (obligatorio para matching)
            query_embedding: Embedding de la consulta (no usado en esta implementación)
            top_k: Número de resultados a retornar
        
        Returns:
            Diccionario con formato compatible con ChromaDB
        """
        if not self.intents.get("intents"):
            return {'distances': [[]], 'metadatas': [[]]}
        
        if query_text is None or query_text == "":
            # Sin texto, no podemos hacer matching
            return {'distances': [[]], 'metadatas': [[]]}
        
        query_lower = query_text.lower().strip()
        results = []
        distances = []
        
        # Buscar en todos los intents
        for intent in self.intents["intents"]:
            tag = intent.get("tag", "").lower()
            patterns = [p.lower() for p in intent.get("patterns", [])]
            
            # 1. Verificar si la query contiene el tag
            tag_match = tag in query_lower if tag else False
            
            # 2. Verificar si coincide con algún patrón
            pattern_match = False
            for pattern in patterns:
                if pattern in query_lower or query_lower in pattern:
                    pattern_match = True
                    break
            
            # 3. Verificar palabras clave comunes
            keywords_match = False
            common_keywords = {
                "saludo": ["hola", "buenos", "buenas", "saludos", "qué tal", "cómo estás"],
                "despedida": ["adiós", "hasta luego", "chao", "bye", "nos vemos"],
                "ayuda": ["ayuda", "ayúdame", "asistencia", "soporte"],
                "gracias": ["gracias", "agradecido", "agradezco"],
            }
            
            for keyword_list in common_keywords.values():
                if any(keyword in query_lower for keyword in keyword_list):
                    keywords_match = True
                    break
            
            # Si hay algún match
            if tag_match or pattern_match or keywords_match:
                # Calcular "distancia" (0 = perfect match, 1 = no match)
                if pattern_match:  # Coincidencia exacta con patrón
                    distance = 0.1
                elif tag_match:    # Coincidencia con tag
                    distance = 0.3
                else:              # Coincidencia con keywords
                    distance = 0.5
                
                results.append({
                    "tag": intent.get("tag", ""),
                    "responses": intent.get("responses", []),
                    "patterns": intent.get("patterns", []),
                    "context": intent.get("context", ""),
                    "match_type": "pattern" if pattern_match else "tag" if tag_match else "keyword"
                })
                distances.append(distance)
        
        # Ordenar por distancia (menor = mejor match)
        if results:
            sorted_results = sorted(zip(results, distances), key=lambda x: x[1])
            results = [r for r, _ in sorted_results[:top_k]]
            distances = [d for _, d in sorted_results[:top_k]]
        
        return {
            'distances': [distances],
            'metadatas': [results]
        }
    def search_documents(self, query_embedding: np.ndarray, top_k: int = 10, min_similarity: float = 0.2) -> Dict:
        """
        Buscar documentos similares al embedding de consulta.
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
            min_similarity: Umbral mínimo de similitud (0-1)
        
        Returns:
            Diccionario con formato compatible con ChromaDB
        """
        if self.index is None or self.index.ntotal == 0:
            return {
                'documents': [[]],
                'distances': [[]],
                'metadatas': [[]]
            }
        
        # Normalizar embedding para búsqueda L2 (equivalente a cosine)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Buscar en FAISS
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(top_k, self.index.ntotal)
        )
        
        # Formatear resultados con filtrado por similitud
        documents_result = []
        metadatas_result = []
        distances_result = []
        indices_result = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                dist = float(distances[0][i])
                # Embeddings normalizados: d² = 2 - 2·cos  →  cos = 1 - d²/2.
                # (Antes se comparaba la distancia L2 directamente, descartando
                #  los chunks MÁS similares; bug corregido.)
                cosine = 1.0 - (dist / 2.0)
                if cosine >= min_similarity:
                    documents_result.append(self.documents[idx])
                    metadatas_result.append(self.metadata[idx])
                    distances_result.append(dist)
                    indices_result.append(int(idx))

        # Si no hay resultados con suficiente similitud, devolver vacío
        if not documents_result:
            return {
                'documents': [[]],
                'distances': [[]],
                'metadatas': [[]],
                'indices': [[]],
            }

        return {
            'documents': [documents_result],
            'distances': [distances_result],
            'metadatas': [metadatas_result],
            'indices': [indices_result],
        }
    
    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Búsqueda semántica con resultados formateados.
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados
        
        Returns:
            Lista de resultados con score de similitud
        """
        results = self.search_documents(query_embedding, top_k)
        indices = results.get('indices', [[]])
        indices = indices[0] if indices else []

        formatted = []
        for pos, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Convertir distancia L2 a similitud coseno (aproximado)
            similarity = 1 / (1 + dist) if dist > 0 else 1.0

            formatted.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity,
                "score": similarity,  # Para compatibilidad
                "doc_index": indices[pos] if pos < len(indices) else None,
            })

        return formatted
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del almacén"""
        return {
            **self.stats,
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "index_type": "FAISS-FlatL2"
        }
    
    def clear(self):
        """Limpiar todos los datos"""
        self.index = None
        self.documents = []
        self.metadata = []
        self.intents = {"intents": []}
        self.doc_id_to_idx = {}
        
        # Eliminar archivos
        for path in [self.index_path, self.documents_path, self.metadata_path, self.intents_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        self.stats = {"total_documents": 0, "last_updated": None}
        logger.info("Almacén vectorial limpiado")