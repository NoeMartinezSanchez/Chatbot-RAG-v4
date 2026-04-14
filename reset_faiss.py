#!/usr/bin/env python
"""
Script para resetear completamente la base de datos FAISS y archivos relacionados.
Útil cuando quieres empezar de cero con una nueva ingesta de documentos.
"""

import os
import shutil
from pathlib import Path
import sys
import json
from datetime import datetime

# Colores para la terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(msg, color=Colors.BLUE):
    """Imprime mensaje con color"""
    print(f"{color}{msg}{Colors.ENDC}")

def reset_faiss_database(backup: bool = True, force: bool = False):
    """
    Elimina los archivos de la base de datos vectorial FAISS
    
    Args:
        backup: Si True, crea una copia de seguridad antes de eliminar
        force: Si True, no pide confirmación
    """
    
    # Definir rutas (basado en tu estructura)
    base_path = Path(__file__).parent
    vector_store_path = base_path / "data" / "vector_store"
    
    # Archivos a eliminar/limpiar
    files_to_handle = [
        vector_store_path / "faiss_index.bin",  # El índice FAISS
        vector_store_path / "documents.pkl",    # Documentos almacenados
        vector_store_path / "metadata.pkl",     # Metadatos
    ]
    
    # Archivos de reporte (opcional)
    report_files = [
        base_path / "data" / "chunks_import_report.json",
        base_path / "data" / "rag_import_report.json",
    ]
    
    print_color("╔══════════════════════════════════════════════════════════╗", Colors.HEADER)
    print_color("║     🗑️  RESET DE BASE DE DATOS FAISS - ChatBot RAG       ║", Colors.HEADER)
    print_color("╚══════════════════════════════════════════════════════════╝", Colors.HEADER)
    
    print(f"\n📂 Directorio target: {vector_store_path}")
    
    # Verificar qué archivos existen
    existing_files = []
    for file_path in files_to_handle:
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            existing_files.append((file_path, size_kb))
    
    if not existing_files:
        print_color("\n✅ No se encontraron archivos de base de datos FAISS.", Colors.GREEN)
        print("   La base de datos ya está vacía.")
        return
    
    # Mostrar archivos que se eliminarán
    print_color("\n📋 Archivos que serán eliminados:", Colors.WARNING)
    for file_path, size_kb in existing_files:
        print(f"   • {file_path.name} ({size_kb:.2f} KB)")
    
    total_size = sum(size_kb for _, size_kb in existing_files)
    print(f"\n📊 Total a eliminar: {total_size:.2f} KB")
    
    # Confirmación (a menos que force=True)
    if not force:
        print_color("\n⚠️  ¿Estás SEGURO de querer eliminar la base de datos?", Colors.WARNING)
        response = input("   Escribe 'BORRAR' para confirmar: ")
        if response != "BORRAR":
            print_color("\n❌ Operación cancelada.", Colors.FAIL)
            return
    
    # Crear backup si se solicita
    if backup:
        backup_dir = base_path / "data" / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        print_color(f"\n💾 Creando backup en: {backup_dir}", Colors.BLUE)
        
        for file_path, _ in existing_files:
            if file_path.exists():
                dest_path = backup_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                print(f"   • Backup de {file_path.name} creado")
        
        print_color("   ✅ Backup completado", Colors.GREEN)
    
    # Eliminar archivos
    print_color("\n🗑️  Eliminando archivos...", Colors.WARNING)
    for file_path, _ in existing_files:
        if file_path.exists():
            file_path.unlink()
            print(f"   • {file_path.name} eliminado")
    
    # También eliminar los reportes (opcional, preguntar)
    print_color("\n📄 ¿Deseas eliminar también los reportes de importación?", Colors.BLUE)
    delete_reports = input("   (s/n): ").lower() == 's'
    
    if delete_reports:
        for report_path in report_files:
            if report_path.exists():
                report_path.unlink()
                print(f"   • {report_path.name} eliminado")
    
    # Verificar resultado
    all_clean = True
    for file_path in files_to_handle:
        if file_path.exists():
            all_clean = False
            print_color(f"   ❌ {file_path.name} aún existe", Colors.FAIL)
    
    if all_clean:
        print_color("\n✅ ¡Base de datos FAISS eliminada completamente!", Colors.GREEN)
        print("\n   Ahora puedes ejecutar:")
        print("   1. python run_chunking.py  (para generar nuevos chunks)")
        print("   2. python scripts/load_chunks_to_rag.py  (para recargar la base)")
    else:
        print_color("\n⚠️  Algunos archivos no pudieron ser eliminados.", Colors.WARNING)

def reset_and_create_empty():
    """
    Versión más radical: elimina y crea un índice FAISS vacío
    """
    from config.settings import FAISS_CONFIG
    import faiss
    import pickle
    
    base_path = Path(__file__).parent
    vector_store_path = base_path / "data" / "vector_store"
    index_path = vector_store_path / "faiss_index.bin"
    
    # Primero resetear
    reset_faiss_database(backup=True, force=False)
    
    # Luego crear índice vacío
    print_color("\n🔄 Creando nuevo índice FAISS vacío...", Colors.BLUE)
    
    # Obtener dimensión de la configuración o usar valor por defecto
    try:
        from config.settings import FAISS_CONFIG
        dimension = FAISS_CONFIG.get("embedding_dim", 384)
    except:
        dimension = 384  # Valor por defecto para paraphrase-multilingual-MiniLM-L12-v2
    
    # Crear índice vacío
    index = faiss.IndexFlatL2(dimension)
    faiss.write_index(index, str(index_path))
    
    # Crear archivos vacíos de documentos y metadatos
    with open(vector_store_path / "documents.pkl", 'wb') as f:
        pickle.dump([], f)
    
    with open(vector_store_path / "metadata.pkl", 'wb') as f:
        pickle.dump([], f)
    
    print_color(f"✅ Nuevo índice FAISS vacío creado en: {index_path}", Colors.GREEN)
    print(f"   • Dimensión: {dimension}")
    print(f"   • Tipo: IndexFlatL2")
    print(f"   • Vectores: 0")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resetear base de datos FAISS")
    parser.add_argument('--force', '-f', action='store_true', 
                       help='Forzar eliminación sin confirmación')
    parser.add_argument('--no-backup', '-nb', action='store_true',
                       help='No crear backup antes de eliminar')
    parser.add_argument('--create-empty', '-c', action='store_true',
                       help='Crear índice vacío después de eliminar')
    
    args = parser.parse_args()
    
    if args.create_empty:
        reset_and_create_empty()
    else:
        reset_faiss_database(backup=not args.no_backup, force=args.force)