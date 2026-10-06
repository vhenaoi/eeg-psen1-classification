"""
check_dependencies.py
=====================
Verifica que todas las dependencias del pipeline están instaladas y
que las rutas de datos son accesibles.

Uso:
    python check_dependencies.py
"""

import sys

OK    = "  [OK]"
FAIL  = "  [FALLA]"
WARN  = "  [AVISO]"

errors = []
warns  = []

print("=" * 55)
print("  VERIFICACIÓN DE DEPENDENCIAS — PORTABLES Pipeline")
print("=" * 55)

# ------------------------------------------------------------------
# 1. Versión de Python
# ------------------------------------------------------------------
print("\n[1] Python")
v = sys.version_info
line = f"  Python {v.major}.{v.minor}.{v.micro}"
if v.major == 3 and v.minor >= 9:
    print(OK, line)
else:
    print(WARN, line, "← recomendado ≥ 3.9")
    warns.append("Python < 3.9")

# ------------------------------------------------------------------
# 2. Paquetes obligatorios
# ------------------------------------------------------------------
print("\n[2] Paquetes obligatorios")

required = [
    ("numpy",           "numpy",          "1.23"),
    ("pandas",          "pandas",         "1.5"),
    ("scipy",           "scipy",          "1.9"),
    ("joblib",          "joblib",         "1.2"),
    ("sklearn",         "scikit-learn",   "1.2"),
    ("imblearn",        "imbalanced-learn","0.10"),
    ("xgboost",         "xgboost",        "1.6"),
    ("matplotlib",      "matplotlib",     "3.6"),
    ("seaborn",         "seaborn",        "0.12"),
    ("pyarrow",         "pyarrow",        "10.0"),   # necesario para leer .feather
    ("openpyxl",        "openpyxl",       "3.0"),    # necesario para Excel output
]

for import_name, pkg_name, min_ver in required:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(OK, f"{pkg_name} {ver}")
    except ImportError:
        print(FAIL, f"{pkg_name}  ← instalar: pip install {pkg_name}>={min_ver}")
        errors.append(pkg_name)

# ------------------------------------------------------------------
# 3. Paquetes opcionales de interpretabilidad
# ------------------------------------------------------------------
print("\n[3] Interpretabilidad (SHAP / SAGE)")

optional = [
    ("shap",  "shap",             "0.41"),
    ("sage",  "sage-importance",  "0.0.4"),
]

for import_name, pkg_name, min_ver in optional:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(OK, f"{pkg_name} {ver}")
    except ImportError:
        print(WARN, f"{pkg_name} no instalado  ← pip install {pkg_name}>={min_ver}")
        print("       (SAGE y SHAP se omitirán pero el resto del pipeline funciona)")
        warns.append(pkg_name)

# ------------------------------------------------------------------
# 4. neuroHarmonize (solo necesario si re-armonizas)
# ------------------------------------------------------------------
print("\n[4] Harmonización (solo si vas a re-correr optional_neuroharmonize.py)")

try:
    import neuroHarmonize
    ver = getattr(neuroHarmonize, "__version__", "?")
    print(OK, f"neuroHarmonize {ver}")
except ImportError:
    print(WARN, "neuroHarmonize no instalado")
    print("       (no necesario para correr experimentos — el feather HARMONIZED ya existe)")

# ------------------------------------------------------------------
# 5. Rutas de datos
# ------------------------------------------------------------------
print("\n[5] Rutas de datos")

import os
try:
    from config import BASE_PATH
    print(OK, f"config.py detectó BASE_PATH:\n       {BASE_PATH}")
except Exception as e:
    print(FAIL, f"config.py falló: {e}")
    errors.append("config.py")
    BASE_PATH = None

if BASE_PATH:
    files_to_check = [
        ("Feather harmonizado",    os.path.join(BASE_PATH, "Resultados", "Data_complete_ce_roi_HARMONIZED.feather")),
        ("PSM 1:1",                os.path.join(BASE_PATH, "Resultados", "PSM_datasets", "Data_matched_ce_roi_PSEN1_1to1.feather")),
        ("PSM 2:1",                os.path.join(BASE_PATH, "Resultados", "PSM_datasets", "Data_matched_ce_roi_PSEN1_2to1.feather")),
        ("PSM 4:1",                os.path.join(BASE_PATH, "Resultados", "PSM_datasets", "Data_matched_ce_roi_PSEN1_4to1.feather")),
        ("PSM 5:1",                os.path.join(BASE_PATH, "Resultados", "PSM_datasets", "Data_matched_ce_roi_PSEN1_5to1.feather")),
        ("Carpeta experiments/",   os.path.join(BASE_PATH, "Resultados", "experiments")),
    ]

    for label, path in files_to_check:
        if os.path.exists(path):
            print(OK, f"{label}")
        else:
            print(FAIL, f"{label}\n       {path}")
            errors.append(label)

# ------------------------------------------------------------------
# 6. Resumen
# ------------------------------------------------------------------
print("\n" + "=" * 55)
if not errors and not warns:
    print("  Todo OK — puedes correr: python run_experiments.py")
elif not errors:
    print(f"  Listo para correr ({len(warns)} aviso(s) menores):")
    for w in warns:
        print(f"    - {w}")
    print("  Comando: python run_experiments.py")
else:
    print(f"  {len(errors)} ERROR(ES) — resuelve antes de correr:")
    for e in errors:
        print(f"    - {e}")
    if warns:
        print(f"  {len(warns)} aviso(s) menor(es): {warns}")
print("=" * 55)
