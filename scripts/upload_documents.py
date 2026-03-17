#!/usr/bin/env python3
"""
SISTEMA DE CARGA DE DOCUMENTOS RAG DESDE EXCEL
Versión profesional para demostración
"""
import os
import sys
import pandas as pd
import json
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.core import RAGSystem
import logging

# Configuración profesional de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExcelRAGLoader:
    """Cargador profesional de documentos Excel al sistema RAG"""
    
    def __init__(self):
        self.rag = RAGSystem()
        self.stats = {
            "total_documents": 0,
            "by_category": {},
            "loaded_at": None
        }
    
    def load_excel_file(self, excel_path: str) -> Dict:
        """
        Carga un archivo Excel estructurado al sistema RAG
        
        Args:
            excel_path: Ruta al archivo Excel (.xlsx)
            
        Returns:
            Dict con estadísticas de carga
        """
        print("=" * 60)
        print("📊 CARGA DE DOCUMENTOS DESDE EXCEL")
        print("=" * 60)
        
        if not os.path.exists(excel_path):
            print(f"❌ ERROR: El archivo {excel_path} no existe")
            return {"error": "Archivo no encontrado"}
        
        try:
            # Cargar el Excel
            print(f"\n📁 Cargando archivo: {os.path.basename(excel_path)}")
            print(f"   📍 Ruta: {excel_path}")
            
            # Leer todas las hojas
            excel_file = pd.ExcelFile(excel_path)
            print(f"   📑 Hojas disponibles: {', '.join(excel_file.sheet_names)}")
            
            # Procesar cada hoja
            total_loaded = 0
            
            for sheet_name in excel_file.sheet_names:
                print(f"\n   📄 Procesando hoja: '{sheet_name}'")
                
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                print(f"      📈 Filas cargadas: {len(df)}")
                
                # Procesar según el tipo de hoja
                if sheet_name.lower() == 'tickets':
                    loaded = self._process_tickets_sheet(df, sheet_name)
                elif 'categoría' in sheet_name.lower():
                    loaded = self._process_categories_sheet(df, sheet_name)
                elif 'respuesta' in sheet_name.lower():
                    loaded = self._process_responses_sheet(df, sheet_name)
                else:
                    loaded = self._process_general_sheet(df, sheet_name)
                
                total_loaded += loaded
                print(f"      ✅ Documentos procesados: {loaded}")
            
            # Actualizar estadísticas
            self.stats["total_documents"] = total_loaded
            self.stats["loaded_at"] = datetime.now().isoformat()
            
            # Generar reporte
            self._generate_report(excel_path)
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error cargando Excel: {e}")
            print(f"\n❌ ERROR CRÍTICO: {e}")
            return {"error": str(e)}
    
    def _process_tickets_sheet(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Procesa la hoja principal de tickets"""
        loaded_count = 0
        
        # Verificar columnas mínimas requeridas
        required_columns = ['Asunto', 'Descripción']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"      ⚠️  Columnas faltantes: {missing_columns}")
            return 0
        
        for idx, row in df.iterrows():
            try:
                # Crear contenido estructurado
                content = self._create_ticket_content(row)
                
                # Crear metadatos enriquecidos
                metadata = self._create_ticket_metadata(row, sheet_name, idx)
                
                # Cargar al RAG
                self.rag.add_document(content, metadata)
                loaded_count += 1
                
                # Actualizar estadísticas por categoría
                categoria = metadata.get('categoria', 'Sin categoría')
                self.stats["by_category"][categoria] = self.stats["by_category"].get(categoria, 0) + 1
                
            except Exception as e:
                logger.warning(f"Error procesando fila {idx}: {e}")
                print(f"      ⚠️  Error en fila {idx}: {str(e)[:50]}...")
        
        return loaded_count
    
    def _create_ticket_content(self, row: pd.Series) -> str:
        """Crea contenido estructurado para un ticket"""
        content_parts = []
        
        # Título
        if 'Asunto' in row and pd.notna(row['Asunto']):
            content_parts.append(f"ASUNTO: {row['Asunto']}")
        
        # Folio
        if 'Folio' in row and pd.notna(row['Folio']):
            content_parts.append(f"FOLIO: {row['Folio']}")
        
        # Descripción
        if 'Descripción' in row and pd.notna(row['Descripción']):
            content_parts.append(f"\nDESCRIPCIÓN DEL PROBLEMA:\n{row['Descripción']}")
        
        # Respuesta
        if 'Respuesta Institucional' in row and pd.notna(row['Respuesta Institucional']):
            content_parts.append(f"\nRESPUESTA INSTITUCIONAL:\n{row['Respuesta Institucional']}")
        
        # Información adicional
        additional_info = []
        for col in ['Categoría', 'Subcategoría', 'Prioridad', 'Área Responsable']:
            if col in row and pd.notna(row[col]):
                additional_info.append(f"{col}: {row[col]}")
        
        if additional_info:
            content_parts.append(f"\nINFORMACIÓN ADICIONAL:\n" + "\n".join(additional_info))
        
        return "\n".join(content_parts)
    
    def _create_ticket_metadata(self, row: pd.Series, sheet_name: str, idx: int) -> Dict[str, Any]:
        """Crea metadatos enriquecidos para un ticket"""
        metadata = {
            "title": row.get('Asunto', f'Ticket_{idx}'),
            "source": "excel_import",
            "sheet_name": sheet_name,
            "row_index": idx,
            "imported_at": datetime.now().isoformat()
        }
        
        # Mapear columnas a metadatos
        column_mapping = {
            'Folio': 'folio',
            'Categoría': 'categoria',
            'Subcategoría': 'subcategoria',
            'Prioridad': 'prioridad',
            'Área Responsable': 'area_responsable',
            'SLA': 'sla_horas'
        }
        
        for excel_col, metadata_key in column_mapping.items():
            if excel_col in row and pd.notna(row[excel_col]):
                metadata[metadata_key] = row[excel_col]
        
        return metadata
    
    def _process_categories_sheet(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Procesa hoja de categorías"""
        loaded_count = 0
        
        for idx, row in df.iterrows():
            try:
                content = f"CATEGORÍA: {row.get('Nombre', 'N/A')}\n"
                content += f"DESCRIPCIÓN: {row.get('Descripción', 'Sin descripción')}\n"
                content += f"SLA: {row.get('SLA', 'N/A')} horas\n"
                
                metadata = {
                    "title": f"Categoría_{row.get('ID_Categoría', idx)}",
                    "type": "category_reference",
                    "source": "excel_import",
                    "sheet_name": sheet_name,
                    "categoria_id": row.get('ID_Categoría', idx),
                    "categoria_nombre": row.get('Nombre', ''),
                    "sla_horas": row.get('SLA', None)
                }
                
                self.rag.add_document(content, metadata)
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Error procesando categoría {idx}: {e}")
        
        return loaded_count
    
    def _process_responses_sheet(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Procesa hoja de respuestas estándar"""
        loaded_count = 0
        
        for idx, row in df.iterrows():
            try:
                content = f"RESPUESTA ESTÁNDAR: {row.get('Código', 'N/A')}\n"
                content += f"SITUACIÓN: {row.get('Situación', 'N/A')}\n"
                content += f"RESPUESTA: {row.get('Respuesta', 'N/A')}\n"
                
                if 'Palabras Clave' in row and pd.notna(row['Palabras Clave']):
                    content += f"PALABRAS CLAVE: {row['Palabras Clave']}\n"
                
                metadata = {
                    "title": f"Respuesta_{row.get('Código', idx)}",
                    "type": "standard_response",
                    "source": "excel_import",
                    "sheet_name": sheet_name,
                    "codigo_respuesta": row.get('Código', f'R{idx:03d}'),
                    "palabras_clave": row.get('Palabras Clave', '').split(',') if 'Palabras Clave' in row else []
                }
                
                self.rag.add_document(content, metadata)
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Error procesando respuesta {idx}: {e}")
        
        return loaded_count
    
    def _process_general_sheet(self, df: pd.DataFrame, sheet_name: str) -> int:
        """Procesa hojas generales"""
        loaded_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Crear contenido combinando todas las columnas
                content_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        content_parts.append(f"{col}: {row[col]}")
                
                content = "\n".join(content_parts)
                
                metadata = {
                    "title": f"{sheet_name}_{idx}",
                    "type": "general_document",
                    "source": "excel_import",
                    "sheet_name": sheet_name,
                    "row_index": idx
                }
                
                self.rag.add_document(content, metadata)
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Error procesando fila general {idx}: {e}")
        
        return loaded_count
    
    def _generate_report(self, excel_path: str):
        """Genera un reporte de carga"""
        print("\n" + "=" * 60)
        print("📈 REPORTE DE CARGA COMPLETADO")
        print("=" * 60)
        
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   📂 Archivo fuente: {os.path.basename(excel_path)}")
        print(f"   📄 Total documentos cargados: {self.stats['total_documents']}")
        print(f"   🕐 Fecha de carga: {self.stats['loaded_at']}")
        
        if self.stats["by_category"]:
            print(f"\n📋 DISTRIBUCIÓN POR CATEGORÍA:")
            for categoria, cantidad in self.stats["by_category"].items():
                print(f"   • {categoria}: {cantidad} documentos")
        
        # Generar archivo de reporte
        report_path = "data/rag_import_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📝 Reporte guardado en: {report_path}")
        print("\n✅ CARGA COMPLETADA EXITOSAMENTE")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sistema de Carga RAG desde Excel - Versión Profesional',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --file data/tickets_4.xlsx
  %(prog)s --file documentos/soporte.xlsx --verbose
  %(prog)s --help

Características:
  • Procesa múltiples hojas de Excel
  • Extrae metadatos estructurados
  • Genera reportes detallados
  • Maneja errores robustamente
        """
    )
    
    parser.add_argument(
        '--file', 
        type=str, 
        required=True,
        help='Ruta al archivo Excel (.xlsx)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Mostrar información detallada del proceso'
    )
    
    args = parser.parse_args()
    
    # Crear loader y cargar
    loader = ExcelRAGLoader()
    stats = loader.load_excel_file(args.file)
    
    # Mostrar estadísticas finales
    if "error" not in stats:
        print("\n🎯 EL SISTEMA RAG ESTÁ LISTO PARA:")
        print("   1. Buscar tickets por folio")
        print("   2. Responder consultas por categoría")
        print("   3. Proporcionar respuestas institucionales")
        print("   4. Identificar áreas responsables")
        print("\n💡 Prueba preguntando:")
        print("   • '¿Qué hacer si perdí mi folio de registro?'")
        print("   • '¿Cómo solicito equivalencia de estudios?'")
        print("   • '¿Cuál es el SLA para soporte técnico?'")

if __name__ == "__main__":
    main()