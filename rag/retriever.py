"""
Vector Store usando FAISS en lugar de ChromaDB
"""
import faiss
import numpy as np
import pickle
import json
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from config.settings import settings  # <-- SE AÑADIO ESTA LINEA

logger = logging.getLogger(__name__)

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
    def search_documents(self, query_embedding: np.ndarray, top_k: int = 3) -> Dict:
        """
        Buscar documentos similares al embedding de consulta.
        
        Args:
            query_embedding: Embedding de la consulta
            top_k: Número de resultados a retornar
        
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
        
        # Formatear resultados
        documents_result = []
        metadatas_result = []
        distances_result = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                documents_result.append(self.documents[idx])
                metadatas_result.append(self.metadata[idx])
                distances_result.append(float(distances[0][i]))
        
        return {
            'documents': [documents_result],
            'distances': [distances_result],
            'metadatas': [metadatas_result]
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
        
        formatted = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Convertir distancia L2 a similitud coseno (aproximado)
            similarity = 1 / (1 + dist) if dist > 0 else 1.0
            
            formatted.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity,
                "score": similarity  # Para compatibilidad
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