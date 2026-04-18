import json
import os

def show_results():
    """Muestra los resultados de la evaluación en los logs"""
    
    # Intentar leer el resumen
    summary_paths = ["/tmp/evaluation_summary.json", "logs/evaluation_summary.json"]
    summary = None
    
    for path in summary_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                summary = json.load(f)
            print(f"\n📊 RESULTADOS DE EVALUACIÓN (desde {path})")
            break
    
    if not summary:
        print("❌ No se encontraron resultados de evaluación")
        return
    
    # Mostrar resumen
    print(f"\n📈 TOTAL: {summary.get('total', 0)} preguntas")
    print(f"✅ Correctas: {summary.get('correctas', 0)}")
    print(f"❌ Incorrectas: {summary.get('incorrectas', 0)}")
    print(f"📊 Tasa de éxito: {summary.get('tasa_exito', 0)}%")
    
    # Mostrar por dificultad
    print(f"\n📊 POR DIFICULTAD:")
    for nivel, datos in summary.get('por_dificultad', {}).items():
        print(f"   {nivel.upper()}: {datos.get('correctas', 0)}/{datos.get('total', 0)} ({datos.get('tasa', 0)}%)")
    
    # Mostrar preguntas falladas
    falladas = summary.get('preguntas_falladas', [])
    if falladas:
        print(f"\n❌ PREGUNTAS FALLADAS ({len(falladas)}):")
        for i, p in enumerate(falladas[:10], 1):  # Mostrar primeras 10
            print(f"\n   {i}. {p.get('pregunta', 'N/A')[:80]}...")
            print(f"      Esperado: {p.get('respuesta_esperada', 'N/A')[:60]}")
            print(f"      Recibido: {p.get('respuesta_chatbot', 'N/A')[:60]}")
    
    # Mostrar resultados por pregunta
    resultados = summary.get('resultados', [])
    if resultados:
        print(f"\n📋 RESULTADOS DETALLADOS:")
        for r in resultados:
            estado = "✅" if r.get('correcto') else "❌"
            print(f"   {estado} {r.get('id', 'N/A')}: {r.get('pregunta', 'N/A')[:50]}...")

if __name__ == "__main__":
    show_results()