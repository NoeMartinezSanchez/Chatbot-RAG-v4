"""
Vector Store usando FAISS con búsqueda híbrida y metadata filtering
Optimizado para los chunks especializados del reporte
"""
import faiss
import numpy as np
import pickle
import json
import os
import hashlib
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from logging import getLogger
from datetime import datetime
from collections import defaultdict

try:
    from evaluation.performance_logger import log_retrieval
except ImportError:
    def log_retrieval(*args, **kwargs):
        pass

logger = getLogger('optimized_retriever')


class OptimizedRetriever:
    """
    Retriever optimizado con:
    - Búsqueda semántica FAISS
    - Filtrado por metadata (antes/después)
    - Re-ranking por importance y metadata_boost
    - Query expansion para Gemma 2B
    - Soporte para multi-query retrieval
    """
    
    def __init__(self, vector_store, config: Dict = None):
        """
        Args:
            vector_store: Instancia de VectorStoreFAISS existente
            config: Configuración personalizada
        """
        self.vs = vector_store
        
        # Configuración por defecto (basada en tu reporte)
        self.config = config or {
            "top_k_initial": 10,     # Recuperar más para filtrar
            "top_k_final": 5,         # Entregar los mejores
            "min_similarity": 0.6,    # Umbral base
            "use_metadata_filter": True,
            "use_reranking": True,
            "use_query_expansion": True,
            "use_multi_query": False,  # Para preguntas complejas
            "metadata_boost": {
                "importance": 1.0,
                "chunk_type": {
                    "pregunta": 1.2,
                    "paso": 1.1,
                    "termino": 1.0,
                    "conducta": 1.3,
                    "regla": 1.1,
                    "articulo": 1.05
                },
                "severity": {
                    "muy_grave": 1.5,
                    "grave": 1.2,
                    "moderado": 1.0,
                    "leve": 0.8
                },
                "action_type": {
                    "obligacion": 1.2,
                    "prohibicion": 1.3,
                    "recomendacion": 1.0
                }
            }
        }
        
        logger.info(f"OptimizedRetriever inicializado con config: {self.config}")
    
    # ==================== 1. INTENT CLASSIFICATION ====================
    
    def classify_intent(self, query: str) -> Dict:
        """
        Clasificar la intención de la query para aplicar filtros específicos.
        Basado en los tipos de documentos de tu reporte.
        """
        query_lower = query.lower()
        
        # Patrones por tipo de documento
        intent_patterns = {
            "normativa": {
                "keywords": ["artículo", "normativa", "reglamento", "control escolar", "art", "capítulo"],
                "doc_types": ["normativa_control_escolar"],
                "chunk_types": ["articulo", "capitulo"],
                "boost": 1.2
            },
            "proceso_inscripcion": {
                "keywords": ["registro", "inscripción", "documentos", "requisitos", "cómo inscribirse", "pasos"],
                "doc_types": ["convocatoria", "guia_aspirante"],
                "chunk_types": ["paso", "requerimiento", "faq"],
                "boost": 1.15
            },
            "conducta_prohibida": {
                "keywords": ["prohibido", "sanción", "falta", "acoso", "hostigamiento", "violencia", "cero tolerancia"],
                "doc_types": ["politica_cero_tolerancia"],
                "severity": ["muy_grave", "grave"],
                "boost": 1.3
            },
            "reglas_comunicacion": {
                "keywords": ["netiqueta", "foro", "mensaje", "comunicación", "virtual", "mayúsculas", "responder"],
                "doc_types": ["reglas_comunicacion_virtual"],
                "chunk_types": ["regla"],
                "boost": 1.1
            },
            "glosario": {
                "keywords": ["qué es", "definición", "significa", "qué significa", "término", "bullying"],
                "chunk_types": ["termino", "definicion"],
                "boost": 1.2
            },
            "protocolo": {
                "keywords": ["protocolo", "convivencia", "objetivo", "ámbito", "cultura de paz"],
                "doc_types": ["protocolo_convivencia"],
                "boost": 1.1
            },
            "decalogo": {
                "keywords": ["principio", "decalogo", "convivencia", "escucha activa", "respeto"],
                "doc_types": ["decalogo_convivencia"],
                "chunk_types": ["principio"],
                "boost": 1.1
            },
            "plazos_fechas": {
                "keywords": ["plazo", "fecha", "cuándo", "vigencia", "término", "días"],
                "has_dates": True,
                "boost": 1.15
            }
        }
        
        # Detectar intención
        matched_intents = []
        for intent_name, patterns in intent_patterns.items():
            score = 0
            keywords_matched = sum(1 for kw in patterns.get("keywords", []) if kw in query_lower)
            if keywords_matched > 0:
                score = keywords_matched / len(patterns.get("keywords", [1]))
            
            if score > 0:
                matched_intents.append({
                    "intent": intent_name,
                    "score": score,
                    "filters": {
                        "doc_types": patterns.get("doc_types", []),
                        "chunk_types": patterns.get("chunk_types", []),
                        "severity": patterns.get("severity", []),
                        "has_dates": patterns.get("has_dates", False)
                    },
                    "boost": patterns.get("boost", 1.0)
                })
        
        # Ordenar por score
        matched_intents.sort(key=lambda x: x["score"], reverse=True)
        
        primary_intent = matched_intents[0] if matched_intents else {
            "intent": "general",
            "score": 0,
            "filters": {},
            "boost": 1.0
        }
        
        logger.debug(f"Query: '{query[:50]}...' → Intent: {primary_intent['intent']} (score: {primary_intent['score']:.2f})")
        
        return primary_intent
    
    # ==================== 2. QUERY EXPANSION ====================
    
    def expand_query(self, query: str, intent: Dict) -> str:
        """
        Expandir la query para mejorar retrieval.
        Especialmente útil para Gemma 2B.
        """
        if not self.config["use_query_expansion"]:
            return query
        
        expanded_queries = [query]
        
        # Expansión por tipo de intent
        intent_name = intent.get("intent", "general")
        
        if intent_name == "normativa":
            expanded_queries.append(f"artículo normativa control escolar {query}")
            expanded_queries.append(f"reglamento {query} disposición oficial")
            
        elif intent_name == "proceso_inscripcion":
            expanded_queries.append(f"pasos requisitos documentos {query}")
            expanded_queries.append(f"cómo realizar {query} procedimiento")
            
        elif intent_name == "conducta_prohibida":
            expanded_queries.append(f"prohibición sanción falta {query}")
            expanded_queries.append(f"política cero tolerancia {query}")
            
        elif intent_name == "reglas_comunicacion":
            expanded_queries.append(f"netiqueta reglas comunicación virtual {query}")
            expanded_queries.append(f"normas foro mensaje {query}")
            
        elif intent_name == "glosario":
            expanded_queries.append(f"definición término significado {query}")
            
        # Expansión con palabras clave genéricas
        if "plazo" in query.lower() or "fecha" in query.lower():
            expanded_queries.append(f"vigencia término días {query}")
        
        # Combinar todas las expansiones
        expanded_query = " | ".join(expanded_queries[:3])  # Limitar a 3
        
        logger.debug(f"Query expandida: '{expanded_query[:100]}...'")
        
        return expanded_query
    
    # ==================== 3. METADATA FILTERING ====================
    
    def apply_metadata_filter(self, results: List[Dict], intent: Dict) -> List[Dict]:
        """
        Filtrar resultados por metadata según la intención detectada.
        """
        if not self.config["use_metadata_filter"]:
            return results
        
        filters = intent.get("filters", {})
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            # Filtrar por doc_type
            if filters.get("doc_types"):
                doc_type = metadata.get("doc_type", metadata.get("document_type", ""))
                if doc_type not in filters["doc_types"]:
                    include = False
            
            # Filtrar por chunk_type
            if include and filters.get("chunk_types"):
                chunk_type = metadata.get("chunk_type", metadata.get("type", ""))
                if chunk_type not in filters["chunk_types"]:
                    include = False
            
            # Filtrar por severity
            if include and filters.get("severity"):
                severity = metadata.get("severity", "")
                if severity not in filters["severity"]:
                    include = False
            
            # Filtrar por has_dates
            if include and filters.get("has_dates"):
                if not metadata.get("has_dates", False):
                    include = False
            
            if include:
                filtered_results.append(result)
        
        logger.debug(f"Filtrado: {len(results)} → {len(filtered_results)} resultados")
        
        return filtered_results if filtered_results else results  # Si queda vacío, devolver todos
    
    # ==================== 4. RE-RANKING CON METADATA BOOST ====================
    
    def rerank_results(self, results: List[Dict], intent: Dict) -> List[Dict]:
        """
        Re-rankear resultados usando importance y metadata_boost.
        Solo aplica boosts si el campo existe y tiene valor no vacío.
        """
        if not self.config["use_reranking"]:
            return results
        
        boosts = self.config["metadata_boost"]
        intent_boost = intent.get("boost", 1.0)
        
        for result in results:
            metadata = result.get("metadata", {})
            base_score = result.get("similarity", result.get("score", 0.5))
            
            total_boost = 1.0
            
            importance = metadata.get("importance")
            if importance is not None:
                total_boost += (importance - 0.5) * boosts.get("importance", 1.0) * 0.5
            
            chunk_type = metadata.get("chunk_type")
            if chunk_type and chunk_type in boosts.get("chunk_type", {}):
                total_boost *= boosts["chunk_type"][chunk_type]
            
            severity = metadata.get("severity")
            if severity and severity in boosts.get("severity", {}):
                total_boost *= boosts["severity"][severity]
            
            action_type = metadata.get("action_type")
            if action_type and action_type in boosts.get("action_type", {}):
                total_boost *= boosts["action_type"][action_type]
            
            total_boost *= intent_boost
            
            result["original_score"] = base_score
            result["boost"] = total_boost
            result["reranked_score"] = min(base_score * total_boost, 1.0)
        
        results.sort(key=lambda x: x.get("reranked_score", 0), reverse=True)
        
        return results[:self.config["top_k_final"]]
    
    # ==================== 5. MULTI-QUERY RETRIEVAL ====================
    
    def generate_subqueries(self, query: str, intent: Dict) -> List[str]:
        """
        Generar sub-preguntas para consultas complejas.
        Útil para: "Explica TODO el proceso de registro" (necesita 4 chunks)
        """
        if not self.config["use_multi_query"]:
            return [query]
        
        query_lower = query.lower()
        subqueries = [query]
        
        # Detectar preguntas complejas
        complex_indicators = ["todo", "todos", "todas", "completo", "completa", "explica", "detalla", "proceso completo"]
        is_complex = any(indicator in query_lower for indicator in complex_indicators)
        
        if is_complex:
            intent_name = intent.get("intent", "general")
            
            if intent_name == "proceso_inscripcion":
                subqueries.extend([
                    "¿Cuáles son los requisitos?",
                    "¿Cuáles son los pasos?",
                    "¿Qué documentos necesito?",
                    "Plazos importantes"
                ])
            elif intent_name == "normativa":
                subqueries.extend([
                    "artículos importantes",
                    "disposiciones generales",
                    "derechos y obligaciones"
                ])
            elif intent_name == "conducta_prohibida":
                subqueries.extend([
                    "conductas muy graves",
                    "sanciones aplicables",
                    "canales de denuncia"
])
        
        logger.debug(f"Generadas {len(subqueries)} subqueries para consulta compleja")
        
        return subqueries
    
    # ==================== 6. MÉTODO PRINCIPAL DE BÚSQUEDA ====================
    
    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = None) -> List[Dict]:
        """
        Pipeline completo de retrieval optimizado.
        
        Args:
            query: Texto de la consulta
            query_embedding: Embedding de la consulta
            top_k: Número de resultados finales
        
        Returns:
            Lista de chunks recuperados y rerankeados
        """
        start_time = time.time()
        top_k = top_k or self.config["top_k_final"]
        
        # PASO 1: Clasificar intención
        intent = self.classify_intent(query)
        intent_name = intent.get("intent", "unknown") if isinstance(intent, dict) else "unknown"
        
        # PASO 2: Expandir query (opcional)
        expanded_query = self.expand_query(query, intent)
        
        # PASO 3: Búsqueda inicial (top_k_initial)
        initial_top_k = self.config["top_k_initial"]
        results = self.vs.semantic_search(query_embedding, top_k=initial_top_k)
        
        if not results:
            logger.warning(f"No se encontraron resultados para: {query[:50]}...")
            search_time = (time.time() - start_time) * 1000
            log_retrieval(
                query=query,
                results=[],
                search_time_ms=search_time,
                filters={"intent": intent_name},
                intent=intent_name
            )
            return []
        
        # PASO 4: Filtrar por metadata
        results = self.apply_metadata_filter(results, intent)
        
        # PASO 5: Re-ranking con metadata_boost
        results = self.rerank_results(results, intent)
        
        # PASO 6: Si es multi-query y hay pocos resultados, intentar expansión
        if self.config["use_multi_query"] and len(results) < 3:
            subqueries = self.generate_subqueries(query, intent)
            if len(subqueries) > 1:
                all_results = []
                for subq in subqueries[:3]:
                    sub_results = self.vs.semantic_search(query_embedding, top_k=3)
                    all_results.extend(sub_results)
                
                seen_ids = set()
                unique_results = []
                for r in all_results:
                    doc_id = r.get("metadata", {}).get("doc_id", "")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_results.append(r)
                
                results = self.rerank_results(unique_results, intent)[:top_k]
        
        search_time = (time.time() - start_time) * 1000
        
        logger.info(f"Retrieved {len(results)} chunks para query: '{query[:50]}...' ({search_time:.1f}ms)")
        
        log_retrieval(
            query=query,
            results=results,
            search_time_ms=search_time,
            filters={"intent": intent_name},
            intent=intent_name
        )
        
        return results
    
    # ==================== 7. MÉTODOS DE UTILIDAD ====================
    
    def get_chunks_by_metadata(self, metadata_filter: Dict, limit: int = 50) -> List[Dict]:
        """
        Recuperar chunks directamente por metadata (sin búsqueda semántica).
        Útil para debugging o retrieval basado en reglas.
        """
        results = []
        
        for idx, metadata in enumerate(self.vs.metadata):
            match = True
            for key, value in metadata_filter.items():
                if metadata.get(key) != value:
                    match = False
                    break
            
            if match and idx < len(self.vs.documents):
                results.append({
                    "content": self.vs.documents[idx],
                    "metadata": metadata,
                    "doc_index": idx
                })
            
            if len(results) >= limit:
                break
        
        logger.info(f"Recuperados {len(results)} chunks por metadata filter: {metadata_filter}")
        
        return results
    
    def get_stats(self) -> Dict:
        """Estadísticas del retriever optimizado"""
        return {
            "config": self.config,
            "total_chunks": len(self.vs.documents),
            "available_metadata_fields": self._get_available_metadata_fields()
        }
    
    def _get_available_metadata_fields(self) -> Dict:
        """Analizar qué campos de metadata están disponibles"""
        fields = defaultdict(set)
        
        for metadata in self.vs.metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    fields[key].add(str(value)[:50])
        
        return {k: list(v)[:10] for k, v in fields.items()}