# Generar menu.json Localmente

Este documento explica cómo generar el archivo `data/menu.json` desde el Excel local.

## Por qué no subir el Excel a Hugging Face

El archivo Excel (`*.xlsx`) está excluido del repositorio porque:
1. Causa errores en Hugging Face Spaces (archivos binarios)
2. Es más pesado que el JSON resultante
3. Solo se necesita localmente para generar el JSON una vez

## Pasos para generar menu.json

### 1. 确保 tener el Excel

Asegúrate de tener el archivo Excel en:
```
data/Navegación Jerárquica_FER.xlsx
```

### 2. Ejecutar el script

```bash
# Activa tu entorno virtual (si tienes uno)
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Ejecutar el generador
python data/build_menu_json.py
```

O con argumentos personalizados:
```bash
python data/build_menu_json.py --excel "data/Mi_Archivo.xlsx" --output "data/menu.json"
```

### 3. Verificar el resultado

El script mostrará un resumen:
```
📊 Resumen:
   - Categorías: X
   - Subcategorías: Y
   - Preguntas: Z

📁 Archivo generado: D:\...\data\menu.json
```

### 4. Subir al repositorio

```bash
git add data/menu.json
git commit -m "Add generated menu.json"
git push
```

## Verificar que funciona

Después de subir, verifica en Hugging Face que el endpoint `/menu` responde correctamente:

```bash
# Local (con el servidor corriendo)
curl http://localhost:8000/menu

# En Hugging Face
curl https://tuespacio.hf.space/menu
```

Debería responder algo como:
```json
{
  "menu": {
    "Categoría 1": {
      "Subcategoría 1": [
        {"question": "...", "answer": "..."}
      ]
    }
  }
}
```

## Solución de problemas

### Error: "Archivo Excel no encontrado"
- Verifica que el archivo Excel esté en la ruta correcta
- El nombre debe ser exactamente: `Navegación Jerárquica_FER.xlsx`

### Error: "No se generaron categorías"
- Verifica que el Excel tenga datos en las hojas
- Las columnas deben llamarse: `subcategoria`, `solucion`, `respuesta`

### El menú no aparece en producción
- Verifica que `data/menu.json` esté en el repositorio
- Revisa los logs de Hugging Face Spaces
