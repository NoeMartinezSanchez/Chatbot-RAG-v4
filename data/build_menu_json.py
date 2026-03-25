"""
Script para construir el archivo JSON del menú desde un Excel.

Lee todas las hojas del archivo Excel y las convierte en una estructura
jerárquica JSON con: Categoría → Subcategoría → Pregunta → Respuesta.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_column_name(col: str) -> str:
    """
    Normaliza nombres de columnas para coincidir con los esperados.
    
    Args:
        col: Nombre original de la columna
        
    Returns:
        Nombre normalizado en minúsculas sin acentos
    """
    col_lower = col.lower().strip()
    mapping = {
        'categoría': 'categoria',
        'subcategoría': 'subcategoria',
        'asunto': 'asunto',
        'solución': 'solucion',
        'pregunta': 'solucion',
        'respuesta': 'respuesta',
        'url': 'url',
        'formato': 'formato',
        'etiquetas': 'etiquetas',
    }
    return mapping.get(col_lower, col_lower)


def get_column_value(row: pd.Series, possible_names: List[str]) -> Optional[str]:
    """
    Obtiene el valor de una columna probando múltiples nombres posibles.
    
    Args:
        row: Fila del DataFrame
        possible_names: Lista de nombres de columna posibles
        
    Returns:
        Valor de la columna o None si no se encuentra
    """
    for col_name in possible_names:
        # Buscar coincidencia exacta o parcial
        for actual_col in row.index:
            if actual_col.lower().strip() == col_name.lower().strip():
                if pd.notna(row[actual_col]):
                    return str(row[actual_col]).strip()
            # También buscar si contiene el nombre
            if col_name.lower() in actual_col.lower():
                if pd.notna(row[actual_col]):
                    return str(row[actual_col]).strip()
    return None


def extract_additional_columns(row: pd.Series) -> Dict[str, Any]:
    """
    Extrae columnas adicionales como metadatos.
    
    Args:
        row: Fila del DataFrame
        
    Returns:
        Diccionario con columnas adicionales
    """
    main_columns = {'categoria', 'subcategoria', 'subcategoría', 'asunto', 'solucion', 'solución', 'respuesta', 'categoría', 'solución /acción'}
    additional = {}
    
    for col in row.index:
        col_normalized = normalize_column_name(col)
        # Ignorar columnas principales
        is_main = False
        for main in main_columns:
            if main in col_normalized or col_normalized in main:
                is_main = True
                break
        if not is_main and pd.notna(row[col]):
            clean_col = col.strip()
            additional[clean_col] = str(row[col]).strip()
    
    return additional


def build_menu_json(excel_path: str, output_json_path: str) -> bool:
    """
    Lee el archivo Excel y genera el JSON del menú.
    
    Args:
        excel_path: Ruta al archivo Excel
        output_json_path: Ruta donde se guardará el JSON
        
    Returns:
        True si se generó correctamente, False si hubo errores
    """
    if not os.path.exists(excel_path):
        logger.warning(f"⚠️ Archivo Excel no encontrado: {excel_path}")
        return False
    
    try:
        logger.info(f"📊 Leyendo Excel: {excel_path}")
        
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        logger.info(f"📋 Hojas encontradas: {sheet_names}")
        
        menu_structure = {}
        
        for sheet_name in sheet_names:
            try:
                logger.info(f"  → Procesando hoja: {sheet_name}")
                
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if df.empty:
                    logger.warning(f"  ⚠️ Hoja vacía: {sheet_name}")
                    continue
                
                # La categoría puede ser el nombre de la hoja O una columna
                category_name = sheet_name.strip()
                subcategories: Dict[str, List[Dict]] = {}
                
                for idx, row in df.iterrows():
                    # La categoría puede estar en columna "Categoría" o usar el nombre de la hoja
                    category_col = get_column_value(row, ['categoria', 'categoría', 'categoría'])
                    if category_col:
                        category_name = category_col
                    
                    # Subcategoría puede ser "Asunto" 
                    subcategory = get_column_value(row, ['subcategoria', 'subcategoría', 'asunto', 'asunto'])
                    
                    # Pregunta puede estar en "Solución" o "Solución /Acción"
                    question = get_column_value(row, ['solucion', 'solución', 'pregunta', 'solución /acción'])
                    
                    # Respuesta
                    answer = get_column_value(row, ['respuesta'])
                    
                    if not question or not answer:
                        continue
                    
                    subcat_key = subcategory if subcategory else "General"
                    
                    if subcat_key not in subcategories:
                        subcategories[subcat_key] = []
                    
                    question_data = {
                        "question": question,
                        "answer": answer,
                    }
                    
                    additional_cols = extract_additional_columns(row)
                    if additional_cols:
                        question_data["metadata"] = additional_cols
                    
                    subcategories[subcat_key].append(question_data)
                
                if subcategories:
                    menu_structure[category_name] = subcategories
                    total_q = sum(len(v) for v in subcategories.values())
                    logger.info(f"    ✓ Categoría '{category_name}': {len(subcategories)} subcategorías, {total_q} preguntas")
                else:
                    logger.warning(f"    ⚠️ Sin datos válidos: {sheet_name}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error procesando hoja '{sheet_name}': {e}")
                continue
        
        if not menu_structure:
            logger.warning("⚠️ No se generó ninguna categoría")
            return False
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(menu_structure, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ JSON guardado en: {output_json_path}")
        logger.info(f"   Categorías: {len(menu_structure)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error general al generar JSON: {e}")
        return False


def load_menu_json(json_path: str) -> Dict[str, Any]:
    """
    Carga el menú desde el archivo JSON.
    
    Args:
        json_path: Ruta al archivo JSON
        
    Returns:
        Diccionario con la estructura del menú
    """
    if not os.path.exists(json_path):
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error cargando JSON: {e}")
        return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera menu.json desde Excel para el chatbot RAG"
    )
    parser.add_argument(
        "--excel", 
        default="data/Navegación Jerárquica_FER.xlsx",
        help="Ruta al archivo Excel de navegación jerárquica"
    )
    parser.add_argument(
        "--output", 
        default="data/menu.json",
        help="Ruta de salida para el archivo JSON"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("🚀 GENERADOR DE MENÚ JSON")
    logger.info("=" * 50)
    logger.info(f"📄 Excel: {args.excel}")
    logger.info(f"📁 Salida: {args.output}")
    logger.info("=" * 50)
    
    success = build_menu_json(args.excel, args.output)
    
    if success:
        logger.info("=" * 50)
        logger.info("✅ PROCESO COMPLETADO")
        logger.info("=" * 50)
        
        # Mostrar resumen
        menu = load_menu_json(args.output)
        total_cats = len(menu)
        total_subcats = sum(len(v) for v in menu.values())
        total_questions = sum(len(w) for v in menu.values() for w in v.values())
        
        logger.info(f"📊 Resumen:")
        logger.info(f"   - Categorías: {total_cats}")
        logger.info(f"   - Subcategorías: {total_subcats}")
        logger.info(f"   - Preguntas: {total_questions}")
        logger.info("")
        logger.info(f"📁 Archivo generado: {os.path.abspath(args.output)}")
        logger.info("")
        logger.info("💡 Este archivo (menu.json) debe subirse al repositorio.")
        logger.info("   NO subir el archivo Excel (*.xlsx)")
    else:
        logger.error("=" * 50)
        logger.error("❌ PROCESO FALLIDO")
        logger.error("=" * 50)
        logger.error(f"Verifica que el archivo Excel exista en: {args.excel}")