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
        if col_name in row.index and pd.notna(row[col_name]):
            return str(row[col_name]).strip()
    return None


def extract_additional_columns(row: pd.Series) -> Dict[str, Any]:
    """
    Extrae columnas adicionales como metadatos.
    
    Args:
        row: Fila del DataFrame
        
    Returns:
        Diccionario con columnas adicionales
    """
    main_columns = {'categoria', 'subcategoria', 'asunto', 'solucion', 'respuesta'}
    additional = {}
    
    for col in row.index:
        col_normalized = normalize_column_name(col)
        if col_normalized not in main_columns and pd.notna(row[col]):
            additional[col_normalized] = str(row[col]).strip()
    
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
                
                category_name = sheet_name.strip()
                subcategories: Dict[str, List[Dict]] = {}
                
                for idx, row in df.iterrows():
                    subcategory = get_column_value(row, ['subcategoria', 'asunto', 'categoría', 'subcategoría'])
                    question = get_column_value(row, ['solucion', 'pregunta', 'solución', 'pregunta'])
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
                    logger.info(f"    ✓ Categoría '{category_name}': {len(subcategories)} subcategorías")
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
    excel_path = "data/Navegación Jerárquica_FER.xlsx"
    output_path = "data/menu.json"
    
    logger.info("🚀 Iniciando generación de menú JSON")
    
    success = build_menu_json(excel_path, output_path)
    
    if success:
        logger.info("✅ Proceso completado")
        menu = load_menu_json(output_path)
        logger.info(f"Estructura: {json.dumps(menu, ensure_ascii=False, indent=2)[:500]}...")
    else:
        logger.error("❌ Proceso fallido")