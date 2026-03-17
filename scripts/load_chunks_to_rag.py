#!/usr/bin/env python3
"""
SISTEMA DE CARGA DE CHUNKS RAG DESDE JSONL
Carga los chunks generados al sistema RAG
"""
import os
import sys
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# AÑADIR ESTAS LÍNEAS AL PRINCIPIO DEL ARCHIVO
# Obtener la ruta absoluta del directorio raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Subir un nivel desde scripts/
sys.path.insert(0, project_root)  # Añadir al principio del path

print(f"📂 Directorio raíz del proyecto: {project_root}")
print(f"📂 Python path actualizado: {sys.path[0]}")

from rag.core import RAGSystem
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunksRAGLoader:
    """Cargador de chunks JSONL al sistema RAG"""
    
    def __init__(self):
        self.rag = RAGSystem()
        self.stats = {
            "total_chunks": 0,
            "by_type": {},
            "by_source": {},
            "loaded_at": None
        }
    
    def load_chunks_file(self, chunks_path: str) -> Dict:
        """
        Carga archivo JSONL de chunks al sistema RAG
        
        Args:
            chunks_path: Ruta al archivo all_chunks.jsonl
            
        Returns:
            Dict con estadísticas de carga
        """
        print("=" * 60)
        print("📊 CARGA DE CHUNKS AL SISTEMA RAG")
        print("=" * 60)
        
        if not os.path.exists(chunks_path):
            print(f"❌ ERROR: El archivo {chunks_path} no existe")
            return {"error": "Archivo no encontrado"}
        
        try:
            print(f"\n📁 Cargando chunks desde: {chunks_path}")
            
            # Leer archivo JSONL línea por línea
            total_loaded = 0
            line_count = 0
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parsear chunk
                        chunk = json.loads(line)
                        
                        # Extraer información
                        text = chunk.get('text', '')
                        chunk_id = chunk.get('chunk_id', f'chunk_{line_num}')
                        doc_type = chunk.get('doc_type', 'unknown')
                        source_file = chunk.get('source_file', 'unknown')
                        metadata = chunk.get('metadata', {})
                        
                        # Crear contenido enriquecido para RAG
                        content = self._create_rag_content(chunk)
                        
                        # Enriquecer metadatos
                        enriched_metadata = {
                            'chunk_id': chunk_id,
                            'doc_type': doc_type,
                            'source_file': source_file,
                            'page_range': chunk.get('page_range', '1-1'),
                            'original_metadata': metadata,
                            'imported_at': datetime.now().isoformat()
                        }
                        
                        # Cargar al RAG
                        self.rag.add_document(content, enriched_metadata)
                        
                        # Actualizar estadísticas
                        total_loaded += 1
                        self.stats["by_type"][doc_type] = self.stats["by_type"].get(doc_type, 0) + 1
                        self.stats["by_source"][source_file] = self.stats["by_source"].get(source_file, 0) + 1
                        
                        # Mostrar progreso cada 10 chunks
                        if total_loaded % 10 == 0:
                            print(f"   ✓ {total_loaded} chunks cargados...")
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decodificando línea {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error procesando chunk {line_num}: {e}")
                    
                    line_count = line_num
            
            # Actualizar estadísticas finales
            self.stats["total_chunks"] = total_loaded
            self.stats["loaded_at"] = datetime.now().isoformat()
            self.stats["lines_processed"] = line_count
            
            # Generar reporte
            self._generate_report(chunks_path)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error cargando chunks: {e}")
            print(f"\n❌ ERROR CRÍTICO: {e}")
            return {"error": str(e)}
    
    def _create_rag_content(self, chunk: Dict) -> str:
        """Crea contenido enriquecido para RAG"""
        parts = []
        
        # Tipo de documento como prefijo
        doc_type = chunk.get('doc_type', 'documento').upper()
        parts.append(f"[{doc_type}]")
        
        # Título si existe
        metadata = chunk.get('metadata', {})
        if 'title' in metadata:
            parts.append(f"\n## {metadata['title']}")
        elif 'section' in metadata:
            parts.append(f"\n## {metadata['section']}")
        
        # Número de página
        page_range = chunk.get('page_range', '1-1')
        parts.append(f"\n📄 Páginas: {page_range}")
        
        # Contenido principal
        text = chunk.get('text', '').strip()
        if text:
            parts.append(f"\n{text}")
        
        # Metadatos específicos según tipo
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
            if 'step' in metadata:
                specific_metadata.append(f"Paso: {metadata.get('step')}")
            if 'is_faq' in metadata and metadata['is_faq']:
                specific_metadata.append("FAQ")
        
        elif doc_type.lower() == 'politica':
            if 'item_number' in metadata:
                specific_metadata.append(f"Punto {metadata.get('item_number')} de {metadata.get('total_items', '?')}")
        
        elif doc_type.lower() == 'decalogo':
            if 'principle_number' in metadata:
                specific_metadata.append(f"Principio {metadata.get('principle_number')}")
        
        if specific_metadata:
            parts.append(f"\n📌 {', '.join(specific_metadata)}")
        
        # Fuente
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
        print(f"   📄 Líneas procesadas: {self.stats.get('lines_processed', 0)}")
        print(f"   ✅ Total chunks cargados: {self.stats['total_chunks']}")
        print(f"   🕐 Fecha de carga: {self.stats['loaded_at']}")
        
        if self.stats["by_type"]:
            print(f"\n📋 DISTRIBUCIÓN POR TIPO DE DOCUMENTO:")
            for doc_type, cantidad in sorted(self.stats["by_type"].items()):
                porcentaje = (cantidad / self.stats['total_chunks']) * 100
                barra = "█" * int(porcentaje / 5)
                print(f"   • {doc_type:<15} {cantidad:3d} ({porcentaje:5.1f}%) {barra}")
        
        if self.stats["by_source"]:
            print(f"\n📋 DISTRIBUCIÓN POR ARCHIVO FUENTE:")
            for source, cantidad in sorted(self.stats["by_source"].items()):
                print(f"   • {source:<40} {cantidad:3d} chunks")
        
        # Guardar reporte
        report_path = Path("data") / "chunks_import_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📝 Reporte guardado en: {report_path}")
        print("\n✅ CARGA COMPLETADA EXITOSAMENTE")

def verify_rag_system():
    """Verifica que el sistema RAG está funcionando"""
    try:
        rag = RAGSystem()
        
        print("\n🔍 VERIFICACIÓN DEL SISTEMA RAG:")
        print(f"   ✅ Sistema RAG inicializado correctamente")
        
        # Verificar componentes
        if hasattr(rag, 'retriever'):
            print(f"   ✅ Componente retriever presente")
        
        if hasattr(rag, 'vector_store'):
            print(f"   ✅ Componente vector_store presente")
            
            # Obtener estadísticas del índice FAISS
            if hasattr(rag.vector_store, 'index') and rag.vector_store.index is not None:
                total_vectors = rag.vector_store.index.ntotal
                print(f"\n📊 Índice FAISS: {total_vectors} vectores")
            
            if hasattr(rag.vector_store, 'documents'):
                print(f"   Documentos: {len(rag.vector_store.documents)}")
        
        if hasattr(rag, 'generator'):
            print(f"   ✅ Componente generator presente")
        
        if hasattr(rag, 'embeddings'):
            print(f"   ✅ Componente embeddings presente")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error verificando sistema RAG: {e}")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sistema de Carga de Chunks al RAG'
    )
    
    parser.add_argument(
        '--chunks', 
        type=str, 
        default='../Documentos RAG/output/chunks/ready_for_rag/all_chunks.jsonl',
        help='Ruta al archivo all_chunks.jsonl'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Mostrar información detallada'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Solo verificar el sistema RAG sin cargar'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_rag_system()
        return
    
    # Verificar sistema RAG
    if not verify_rag_system():
        print("\n❌ No se puede continuar. Sistema RAG no disponible.")
        return
    
    # Verificar que el archivo existe
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"\n❌ ERROR: No se encuentra el archivo: {chunks_path}")
        print(f"   📍 Ruta absoluta: {chunks_path.absolute()}")
        return
    
    # Cargar chunks
    loader = ChunksRAGLoader()
    stats = loader.load_chunks_file(str(chunks_path))
    
    if "error" not in stats:
        print("\n" + "=" * 60)
        print("🎯 SISTEMA RAG AHORA PUEDE RESPONDER SOBRE:")
        print("=" * 60)
        print("\n📚 DOCUMENTOS CARGADOS:")
        for doc_type, cantidad in stats.get("by_type", {}).items():
            print(f"   • {doc_type.capitalize()}: {cantidad} chunks")
        
        print("\n💡 PRUEBA PREGUNTANDO:")
        print("   • '¿Cuáles son los requisitos de ingreso?'")
        print("   • '¿Qué dice el artículo 1 de las normas?'")
        print("   • '¿Cómo hago mi registro en la convocatoria?'")
        print("   • '¿Qué políticas de cero tolerancia existen?'")

if __name__ == "__main__":
    main()