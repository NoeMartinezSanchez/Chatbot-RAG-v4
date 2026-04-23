"""
Módulo principal del sistema RAG (Retrieval-Augmented Generation)
"""
import logging
from typing import Tuple, Dict, Any, List
import json
import os
import random
import re

from config.settings import settings
from .embeddings import EmbeddingModel
from .retriever import VectorStoreFAISS
from .generator import TinyLlamaGenerator
from .gemma_generator import GemmaGenerator
from .optimized_retriever import OptimizedRetriever

try:
    from evaluation.performance_logger import log_latency
    _LOGGING_ENABLED = True
except ImportError:
    _LOGGING_ENABLED = False
    def log_latency(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStoreFAISS()
        self.generator = GemmaGenerator()
        self.optimized_retriever = OptimizedRetriever(self.vector_store)
        self.intents_loaded = False

        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        logger.info("RAG System initialized con OptimizedRetriever")
    
    def load_intents(self, intents_file: str = "data/vector_store/intents.json"):
        """Carga intents al sistema"""
        try:
            self.vector_store.store_intents(intents_file)
            self.intents_loaded = True
            logger.info("Intents loaded into FAISS vector store")
        except Exception as e:
            logger.error(f"Error loading intents: {e}")
            # Crear archivo básico si no existe
            if not os.path.exists(intents_file):
                basic_intents = {
                    "intents": [
                        {
                            "tag": "saludo",
                            "patterns": ["hola", "buenos días", "buenas tardes"],
                            "responses": ["¡Hola! ¿En qué puedo ayudarte?"],
                            "context": "welcome"
                        }
                    ]
                }
                os.makedirs(os.path.dirname(intents_file), exist_ok=True)
                with open(intents_file, 'w', encoding='utf-8') as f:
                    json.dump(basic_intents, f, ensure_ascii=False, indent=2)
                logger.info("Created basic intents file")
    
    def _clean_query(self, query: str) -> str:
        """
        Limpiar consulta para mejor matching con intents.
        """
        # Quitar signos de puntuación y convertir a minúsculas
        clean = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ¿?]', '', query.lower())
        return clean.strip()
    
    def _classify_query_type(self, query: str) -> str:
        """
        Clasificar el tipo de consulta para decidir prioridad.
        """
        query_lower = query.lower()
        
        # Palabras clave para intents (saludos, despedidas, etc.)
        intent_keywords = {
            'saludo': ['hola', 'buen día', 'buenas', 'saludos', 'qué tal', 'cómo estás'],
            'despedida': ['adiós', 'hasta luego', 'chao', 'bye', 'nos vemos'],
            'gracias': ['gracias', 'agradecido', 'agradezco'],
            'ayuda_general': ['ayuda', 'ayúdame', 'asistencia', 'soporte']
        }
        
        # Verificar si es un intent básico
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent_type
        
        # Preguntas técnicas/complejas van a RAG
        question_words = ['cómo', 'dónde', 'cuándo', 'qué', 'por qué', 'cuál', 'cuánto']
        if any(query_lower.startswith(word) for word in question_words):
            return 'rag_preferido'
        
        return 'neutral'
    
    def _should_use_intent(self, query: str, intent_results: Dict, intent_priority: str) -> bool:
        """
        Decidir si usar intent basado en múltiples criterios.
        """
        # 1. Si es saludo/despedida, SIEMPRE usar intent
        if intent_priority in ['saludo', 'despedida', 'gracias']:
            return True
        
        # 2. Si no hay resultados de intent, usar RAG
        if not intent_results.get('metadatas') or not intent_results['metadatas'][0]:
            return False
        
        # 3. Verificar calidad del match
        best_intent = intent_results['metadatas'][0][0] if intent_results['metadatas'][0] else None
        best_distance = intent_results['distances'][0][0] if intent_results['distances'][0] else 1.0
        
        # 4. Reglas de decisión
        if intent_priority == 'rag_preferido':
            # Preguntas técnicas: solo usar intent si es MUY bueno
            return best_distance < 0.3  # Match muy cercano
        
        # Caso general: balancear longitud y calidad
        if len(query) < 100:  # No demasiado larga
            # Distancia baja = buen match
            if best_distance < 0.5:  # Ajusta según necesidad
                return True
        
        return False
    
    def _format_intent_response(self, intent_results: Dict) -> Tuple[str, bool, float, list]:
        """
        Formatear respuesta de intent para compatibilidad.
        """
        if not intent_results.get('metadatas') or not intent_results['metadatas'][0]:
            return ("", False, 0.0, [])
        
        best_intent = intent_results['metadatas'][0][0]
        best_distance = intent_results['distances'][0][0]
        
        # Convertir distancia a confianza
        confidence = max(0.0, 1.0 - best_distance)
        
        # Seleccionar respuesta aleatoria del intent
        responses = best_intent.get('responses', ['Lo siento, no tengo una respuesta preparada.'])
        response = random.choice(responses)
        
        return (response, False, confidence, [])
    
    def _rag_process(self, query: str) -> Tuple[str, bool, float, list]:
        """
        Procesar consulta usando RAG con OptimizedRetriever.
        """
        import time
        
        try:
            # 1. Generar embedding de la consulta
            query_embedding = self.embedder.embed_text(query)
            
            # 2. Usar OptimizedRetriever para búsqueda avanzada
            results = self.optimized_retriever.retrieve(
                query, 
                query_embedding, 
                top_k=settings.TOP_K_RESULTS
            )
            
            # 3. Verificar si hay resultados relevantes
            if not results:
                return "No encontré información específica sobre eso en los materiales de Prepa en Línea SEP.", False, 0.0, []

            # 4. Extraer contextos y metadatos
            contexts = [r.get("content", r.get("text", "")) for r in results]
            metadatas = [r.get("metadata", {}) for r in results]
            context_str = " ".join(contexts)
            
            if not context_str.strip():
                return "No encontré información específica sobre eso en los materiales de Prepa en Línea SEP.", False, 0.0, []
            
            # Variables para métricas
            generation_start = time.time()
            tokens_generated = 0
            
            # 5. Generar respuesta RAG con Gemma (con callback para métricas)
            def on_tokens(tokens: int, elapsed: float):
                nonlocal tokens_generated
                tokens_generated = tokens
            
            logger.info(f"🔄 Calling generator.generate() with query length: {len(query)}, context length: {len(context_str)}")
            try:
                response = self.generator.generate(query, context_str, on_tokens_generated=on_tokens)
                logger.info(f"✅ Response received: {response[:100]}...")
            except Exception as e:
                logger.error(f"❌ Error in generator.generate(): {e}", exc_info=True)
                raise
            
            generation_time = (time.time() - generation_start) * 1000
            
            # Log de latencia
            if _LOGGING_ENABLED and tokens_generated > 0:
                log_latency(
                    retrieval_time_ms=0,
                    generation_time_ms=generation_time,
                    total_time_ms=generation_time,
                    tokens_generated=tokens_generated,
                    question=query
                )
            
            # 6. Preparar fuentes para mostrar
            sources = []
            for i, (context, metadata) in enumerate(zip(contexts, metadatas)):
                if i < 3 and context:
                    source_info = {
                        "content": context,
                        "metadata": metadata
                    }
                    sources.append(source_info)
            
            # 7. Calcular confianza basada en reranked_score
            confidence = 0.0
            if results and results[0].get("reranked_score"):
                confidence = results[0]["reranked_score"]
            elif results and results[0].get("similarity"):
                confidence = results[0]["similarity"]
            
            return response, True, confidence, sources
            
        except Exception as e:
            logger.error(f"Error en RAG process: {e}")
            return "Lo siento, tuve un problema procesando tu pregunta. ¿Podrías intentarlo de nuevo?", False, 0.0, []
    
    def process_query(self, query: str) -> Tuple[str, bool, float, list]:
        """
        Procesa una consulta y retorna respuesta y metadata
        
        Returns:
            Tuple[str, bool, float, list]: (respuesta, es_rag, confianza, fuentes)
        """
        query = query.strip()
        
        # Solo usar intents para despedidas EXPLÍCITAS
        if self.intents_loaded:
            despedidas = ['adiós', 'adios', 'bye', 'chao', 'nos vemos', 'hasta luego', 'me voy', 'me retiro']
            if any(palabra in query.lower() for palabra in despedidas):
                logger.info(f"Detectada despedida, buscando intent: '{query[:50]}...'")
                intent_results = self.vector_store.search_intents(
                    query_text=query.lower(),
                    top_k=1
                )
                if intent_results.get('metadatas') and intent_results['metadatas'][0]:
                    return self._format_intent_response(intent_results)
        
        # Para TODO lo demás (incluyendo saludos), usar RAG con OptimizedRetriever
        logger.info(f"Usando RAG + OptimizedRetriever para: '{query[:50]}...'")
        return self._rag_process(query)
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """Añade un documento al sistema"""
        if metadata is None:
            metadata = {}
        
        try:
            # Generar embedding
            embedding = self.embedder.embed_text(content)
            
            # Añadir al vector store
            self.vector_store.add_document(content, metadata, embedding)
            
            logger.info(f"Document added: {metadata.get('title', 'No title')}")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def add_documents_batch(self, documents: List[Dict[str, Any]]):
        """Añade múltiples documentos en lote"""
        if not documents:
            return
        
        try:
            # Extraer textos
            texts = [doc['content'] for doc in documents]
            
            # Generar embeddings en batch (usar prefijo de passage para indexar)
            embeddings = self.embedder.embed_batch(texts, is_passage=True)
            
            # Añadir al vector store
            self.vector_store.add_documents(documents, embeddings)
            
            logger.info(f"Added {len(documents)} documents in batch")
            
        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
    
    def get_stats(self):
        """Obtener estadísticas del sistema"""
        try:
            return {
                "vector_store": self.vector_store.get_stats(),
                "embedding_model": self.embedder.model_name,
                "intents_loaded": self.intents_loaded
            }
        except:
            return {"status": "unknown"}
    
    def _simple_extract_response(self, query: str, context: str) -> str:
        """Fallback: Extraer respuesta directamente del contexto sin TinyLlama"""
        import re
        
        # Limpiar el contexto
        lines = context.split('\n')
        clean_lines = []
        for line in lines:
            if re.match(r'^\[.*?\]$', line):
                continue
            if re.match(r'^#{2,}', line):
                continue
            if re.match(r'^📄', line):
                continue
            if re.match(r'^Fila:', line):
                continue
            if re.match(r'^Hoja:', line):
                continue
            if line.strip() and len(line.strip()) > 10:
                clean_lines.append(line.strip())
        
        context_clean = ' '.join(clean_lines)
        
        # Extraer oraciones que contengan palabras clave de la pregunta
        query_words = set(query.lower().split())
        sentences = context_clean.split('. ')
        
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            # Buscar coincidencia de palabras
            matches = sum(1 for w in query_words if w in sent_lower and len(w) > 3)
            if matches > 0:
                relevant.append(sent.strip())
        
        if relevant:
            return '. '.join(relevant[:3]) + '.'
        
        # Si no hay coincidencias, devolver el inicio del contexto
        return context_clean[:300] + '...' if len(context_clean) > 300 else context_clean