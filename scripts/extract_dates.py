"""Extractor automático de fechas desde documentos"""
import re
from datetime import datetime
from typing import List, Dict


class DateExtractor:
    """Extrae fechas automáticamente de texto"""

    def __init__(self):
        self.meses = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }

    def extract_dates(self, text: str) -> List[Dict]:
        """Extrae todas las fechas del texto"""
        fechas = []

        patron1 = r'(\d{1,2}) de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?: de (\d{4}))?'

        for match in re.finditer(patron1, text, re.IGNORECASE):
            dia = int(match.group(1))
            mes = self.meses[match.group(2).lower()]
            año = int(match.group(3)) if match.group(3) else datetime.now().year

            fechas.append({
                'fecha': f"{año}-{mes:02d}-{dia:02d}",
                'texto_original': match.group(0),
                'tipo': 'fecha'
            })

        patron2 = r'del (\d{1,2}) de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre) al (\d{1,2}) de (enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)'

        for match in re.finditer(patron2, text, re.IGNORECASE):
            dia_inicio = int(match.group(1))
            mes_inicio = self.meses[match.group(2).lower()]
            dia_fin = int(match.group(3))
            mes_fin = self.meses[match.group(4).lower()]
            año = datetime.now().year

            fechas.append({
                'fecha_inicio': f"{año}-{mes_inicio:02d}-{dia_inicio:02d}",
                'fecha_fin': f"{año}-{mes_fin:02d}-{dia_fin:02d}",
                'texto_original': match.group(0),
                'tipo': 'rango'
            })

        return fechas

    def comparar_con_hoy(self, fecha_str: str) -> str:
        """Compara una fecha con hoy y devuelve: 'pasado', 'hoy', 'futuro'"""
        fecha = datetime.fromisoformat(fecha_str)
        hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        fecha = fecha.replace(hour=0, minute=0, second=0, microsecond=0)

        if fecha < hoy:
            return "pasado"
        elif fecha == hoy:
            return "hoy"
        else:
            return "futuro"


if __name__ == "__main__":
    extractor = DateExtractor()

    texto = """
    La convocatoria estará abierta del 26 de enero al 4 de febrero de 2026.
    El módulo propedéutico comienza el 15 de marzo.
    """

    fechas = extractor.extract_dates(texto)
    print("Fechas encontradas:")
    for f in fechas:
        print(f"  - {f}")
        if f['tipo'] == 'rango':
            print(f"    Estado: {extractor.comparar_con_hoy(f['fecha_inicio'])}")
