"""
config.py — Configuración de rutas del proyecto PORTABLES.

ÚNICA ruta a cambiar si los datos se mueven a una ubicación nueva.
Los candidatos se prueban en orden; se usa el primero que exista.
"""

import os

_CANDIDATES = [
    # Unidad compartida — vista desde PC propio (cuenta Sapienza)
    r'I:\Unidades compartidas\Sapienza - Verónica Henao Isaza\PERSONAL\PORTABLES',
    # Unidad compartida — cuenta Gmail (fallback si I: no está montada)
    r'H:\Unidades compartidas\Sapienza - Verónica Henao Isaza\PERSONAL\PORTABLES',
    # Unidad compartida — vista desde PC universitario
    r'G:\Drive condivisi\Sapienza - Verónica Henao Isaza\PERSONAL\PORTABLES',
    # Disco local (laptop/casa)
    r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES',
]

BASE_PATH = next((p for p in _CANDIDATES if os.path.isdir(p)), _CANDIDATES[0])

if not os.path.isdir(BASE_PATH):
    raise RuntimeError(
        f"No se encontró la carpeta PORTABLES en ninguna ruta conocida.\n"
        f"Rutas probadas:\n" + "\n".join(f"  {p}" for p in _CANDIDATES) +
        "\n\nEdita config.py y añade la ruta correcta a _CANDIDATES."
    )
