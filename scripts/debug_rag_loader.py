# scripts/debug_rag_loader.py
"""
Script de diagnóstico para el cargador RAG
"""
import sys
import os
import traceback

print("=" * 60)
print("🔍 DIAGNÓSTICO DEL SISTEMA RAG")
print("=" * 60)

# 1. Información del sistema
print(f"\n📁 Directorio actual: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")
print(f"📂 Python path:")

for i, path in enumerate(sys.path, 1):
    print(f"   {i}. {path}")

# 2. Verificar estructura de carpetas
print("\n📂 Verificando estructura de rag/:")
rag_path = os.path.join(os.getcwd(), "rag")
if os.path.exists(rag_path):
    print(f"   ✅ carpeta rag/ existe")
    rag_files = os.listdir(rag_path)
    print(f"   Archivos en rag/: {rag_files}")
else:
    print(f"   ❌ carpeta rag/ NO existe")

# 3. Intentar importaciones paso a paso
print("\n📦 Probando importaciones:")

try:
    print("   Intentando importar rag.core...")
    from rag.core import RAGSystem
    print("   ✅ rag.core importado")
except Exception as e:
    print(f"   ❌ Error importando rag.core: {e}")
    traceback.print_exc()

try:
    print("\n   Intentando importar rag.retriever...")
    from rag.retriever import VectorStoreFAISS
    print("   ✅ rag.retriever importado")
except Exception as e:
    print(f"   ❌ Error importando rag.retriever: {e}")

try:
    print("\n   Intentando importar rag.embeddings...")
    from rag.embeddings import EmbeddingModel
    print("   ✅ rag.embeddings importado")
except Exception as e:
    print(f"   ❌ Error importando rag.embeddings: {e}")

try:
    print("\n   Intentando importar rag.generator...")
    from rag.generator import ResponseGenerator
    print("   ✅ rag.generator importado")
except Exception as e:
    print(f"   ❌ Error importando rag.generator: {e}")

# 4. Inicializar RAGSystem
print("\n🔄 Inicializando RAGSystem...")
try:
    rag = RAGSystem()
    print("✅ RAGSystem inicializado correctamente")
    
    # Verificar componentes
    print("\n🔍 Componentes disponibles:")
    print(f"   - embeddings: {'✅' if hasattr(rag, 'embeddings') else '❌'}")
    print(f"   - retriever: {'✅' if hasattr(rag, 'retriever') else '❌'}")
    print(f"   - generator: {'✅' if hasattr(rag, 'generator') else '❌'}")
    print(f"   - vector_store: {'✅' if hasattr(rag, 'vector_store') else '❌'}")
    
    if hasattr(rag, 'retriever'):
        print("\n📋 Métodos del retriever:")
        retriever_methods = [m for m in dir(rag.retriever) if not m.startswith('_')]
        for method in retriever_methods:
            print(f"   - {method}")
    
    if hasattr(rag, 'vector_store'):
        print("\n📊 Estadísticas del vector_store:")
        try:
            stats = rag.vector_store.get_stats()
            for key, value in stats.items():
                print(f"   - {key}: {value}")
        except:
            print("   No se pudieron obtener estadísticas")
            
except Exception as e:
    print(f"❌ Error inicializando RAGSystem: {e}")
    traceback.print_exc()

# 5. Verificar archivo de chunks
print("\n📁 Verificando archivo de chunks:")
chunks_path = os.path.join("..", "Documentos RAG", "output", "chunks", "ready_for_rag", "all_chunks.jsonl")
print(f"   Ruta: {chunks_path}")
print(f"   Existe: {'✅' if os.path.exists(chunks_path) else '❌'}")

if os.path.exists(chunks_path):
    file_size = os.path.getsize(chunks_path) / 1024  # KB
    print(f"   Tamaño: {file_size:.1f} KB")
    
    # Mostrar primeras líneas
    print(f"\n   📝 Primeras líneas del archivo:")
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i <= 3:  # Mostrar solo las primeras 3 líneas
                    preview = line[:150] + "..." if len(line) > 150 else line
                    print(f"      Línea {i}: {preview}")
                else:
                    break
    except Exception as e:
        print(f"   Error leyendo archivo: {e}")

print("\n" + "=" * 60)
print("✅ DIAGNÓSTICO COMPLETADO")
print("=" * 60)