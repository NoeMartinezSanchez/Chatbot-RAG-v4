#!/usr/bin/env python3
"""
SISTEMA DE CARGA DE CHUNKS RAG DESDE JSONL
Carga los chunks generados al vector store FAISS (sin generador)
"""
import os
import sys
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"📂 Directorio raíz: {project_root}")

from rag.embeddings import EmbeddingModel
from rag.retriever import VectorStoreFAISS
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunksRAGLoader:
    """Cargador de chunks JSONL al vector store FAISS"""
    
    def __init__(self):
        print("🔄 Inicializando embeddings...")
        self.embeddings = EmbeddingModel()
        self.vector_store = VectorStoreFAISS()
        print(f"   ✅ VectorStoreFAISS inicializado")
        
        self.stats = {
            "total_chunks": 0,
            "by_type": {},
            "by_source": {},
            "loaded_at": None
        }
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de textos"""
        return self.embeddings.embed_batch(texts)
    
    def load_chunks_file(self, chunks_path: str) -> Dict:
        """Carga archivo JSONL de chunks al vector store"""
        print("=" * 60)
        print("📊 CARGA DE CHUNKS AL VECTOR STORE")
        print("=" * 60)
        
        if not os.path.exists(chunks_path):
            print(f"❌ ERROR: El archivo {chunks_path} no existe")
            return {"error": "Archivo no encontrado"}
        
        try:
            print(f"\n📁 Cargando chunks desde: {chunks_path}")
            
            # Primero, leer todos los chunks
            chunks_list = []
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        chunks_list.append(chunk)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decodificando línea {line_num}: {e}")
            
            print(f"   📄 Total chunks encontrados: {len(chunks_list)}")
            
            if not chunks_list:
                return {"error": "No se encontraron chunks válidos"}
            
            # Preparar documentos y embeddings
            documents = []
            embeddings_list = []
            texts_for_embedding = []
            
            for chunk in chunks_list:
                text = chunk.get('text', '')
                if not text:
                    continue
                
                doc_type = chunk.get('doc_type', 'unknown')
                source_file = chunk.get('source_file', 'unknown')
                metadata = chunk.get('metadata', {})
                chunk_id = chunk.get('chunk_id', 'unknown')
                
                # Crear contenido enriquecido
                content = self._create_rag_content(chunk)
                texts_for_embedding.append(content)
                
                # Preparar documento para FAISS
                doc = {
                    'content': content,
                    'metadata': {
                        'chunk_id': chunk_id,
                        'doc_type': doc_type,
                        'source_file': source_file,
                        'page_range': chunk.get('page_range', '1-1'),
                        'original_metadata': metadata,
                        'imported_at': datetime.now().isoformat()
                    }
                }
                documents.append(doc)
                
                # Actualizar estadísticas
                self.stats["by_type"][doc_type] = self.stats["by_type"].get(doc_type, 0) + 1
                self.stats["by_source"][source_file] = self.stats["by_source"].get(source_file, 0) + 1
            
            print(f"   🧠 Generando embeddings para {len(texts_for_embedding)} textos...")
            
            # Generar embeddings en batches para eficiencia
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts_for_embedding), batch_size):
                batch = texts_for_embedding[i:i+batch_size]
                batch_embeddings = self._generate_embeddings_batch(batch)
                all_embeddings.append(batch_embeddings)
                print(f"      Batch {i//batch_size + 1}: {len(batch)} embeddings generados")
            
            embeddings = np.vstack(all_embeddings)
            print(f"   ✅ {len(embeddings)} embeddings generados (dimensión: {embeddings.shape[1]})")
            
            # Añadir al vector store
            print(f"   💾 Añadiendo documentos al índice FAISS...")
            self.vector_store.add_documents(documents, embeddings)
            
            self.stats["total_chunks"] = len(documents)
            self.stats["loaded_at"] = datetime.now().isoformat()
            
            self._generate_report(chunks_path)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error cargando chunks: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _create_rag_content(self, chunk: Dict) -> str:
        """Crea contenido enriquecido para el vector store"""
        parts = []
        
        doc_type = chunk.get('doc_type', 'documento').upper()
        parts.append(f"[{doc_type}]")
        
        metadata = chunk.get('metadata', {})
        if 'title' in metadata:
            parts.append(f"\n## {metadata['title']}")
        elif 'section' in metadata:
            parts.append(f"\n## {metadata['section']}")
        
        page_range = chunk.get('page_range', '1-1')
        parts.append(f"\n📄 Páginas: {page_range}")
        
        text = chunk.get('text', '').strip()
        if text:
            parts.append(f"\n{text}")
        
        # Metadatos específicos según tipo de documento
        specific_metadata = []
        
        if doc_type.lower() == 'convocatoria':
            if 'section_number' in metadata:
                specific_metadata.append(f"Sección: {metadata.get('section_number')}")
        elif doc_type.lower() == 'normativa':
            if 'article' in metadata:
                specific_metadata.append(f"Artículo: {metadata.get('article')}")
            if 'chapter' in metadata:
                specific_metadata.append(f"Capítulo: {metadata.get('chapter')}")
        elif doc_type.lower() == 'guia':
            if 'step_number' in metadata:
                specific_metadata.append(f"Paso: {metadata.get('step_number')}")
        elif doc_type.lower() == 'decalogo':
            if 'principle_number' in metadata:
                specific_metadata.append(f"Principio {metadata.get('principle_number')}")
        elif doc_type.lower() == 'politica_cero_tolerancia':
            if 'conduct_name' in metadata:
                specific_metadata.append(f"Conducta: {metadata.get('conduct_name')}")
        elif doc_type.lower() == 'protocolo':
            if 'term' in metadata:
                specific_metadata.append(f"Término: {metadata.get('term')}")
        elif doc_type.lower() == 'reglas_comunicacion':
            if 'rule_title' in metadata:
                specific_metadata.append(f"Regla: {metadata.get('rule_title')}")
        
        if specific_metadata:
            parts.append(f"\n📌 {', '.join(specific_metadata)}")
        
        source_file = chunk.get('source_file', 'Documento oficial')
        parts.append(f"\n📚 Fuente: {source_file}")
        
        return "\n".join(parts)
    
    def _generate_report(self, chunks_path: str):
        """Genera un reporte de carga"""
        print("\n" + "=" * 60)
        print("📈 REPORTE DE CARGA COMPLETADO")
        print("=" * 60)
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   📂 Archivo fuente: {os.path.basename(chunks_path)}")
        print(f"   ✅ Total chunks cargados: {self.stats['total_chunks']}")
        print(f"   🕐 Fecha de carga: {self.stats['loaded_at']}")
        
        if self.stats["by_type"]:
            print(f"\n📋 DISTRIBUCIÓN POR TIPO DE DOCUMENTO:")
            for doc_type, cantidad in sorted(self.stats["by_type"].items()):
                porcentaje = (cantidad / self.stats['total_chunks']) * 100 if self.stats['total_chunks'] > 0 else 0
                barra = "█" * int(porcentaje / 5)
                print(f"   • {doc_type:<35} {cantidad:3d} ({porcentaje:5.1f}%) {barra}")
        
        if self.stats["by_source"]:
            print(f"\n📋 DISTRIBUCIÓN POR ARCHIVO FUENTE:")
            for source, cantidad in sorted(self.stats["by_source"].items()):
                print(f"   • {source[:60]:<60} {cantidad:3d} chunks")
        
        # Mostrar estado del índice FAISS
        faiss_stats = self.vector_store.get_stats()
        print(f"\n📊 ESTADO DEL ÍNDICE FAISS:")
        print(f"   • Total vectores: {faiss_stats.get('index_size', 0)}")
        print(f"   • Dimensión: {faiss_stats.get('embedding_dim', 'N/A')}")
        print(f"   • Directorio persistencia: {self.vector_store.persist_directory}")
        
        print("\n✅ CARGA COMPLETADA EXITOSAMENTE")
        
        print("\n" + "=" * 60)
        print("🎯 VECTOR STORE ACTUALIZADO")
        print("=" * 60)
        print(f"\n📚 Total documentos en índice: {self.stats['total_chunks']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sistema de Carga de Chunks al Vector Store FAISS'
    )
    
    parser.add_argument(
        '--chunks', 
        type=str, 
        default='../Documentos RAG/output/chunks/ready_for_rag/all_chunks.jsonl',
        help='Ruta al archivo all_chunks.jsonl'
    )
    
    args = parser.parse_args()
    
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"\n❌ ERROR: No se encuentra el archivo: {chunks_path}")
        print(f"   📍 Ruta absoluta: {chunks_path.absolute()}")
        return
    
    loader = ChunksRAGLoader()
    stats = loader.load_chunks_file(str(chunks_path))
    
    if "error" not in stats and stats.get('total_chunks', 0) > 0:
        print(f"\n✅ ÉXITO: {stats['total_chunks']} chunks cargados")
    elif "error" in stats:
        print(f"\n❌ Error: {stats['error']}")


if __name__ == "__main__":
    main()