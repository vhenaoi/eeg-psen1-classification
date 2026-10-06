"""
config.py -- paths for the Frontiers PSEN1 re-run (pipeline_v3).

BASE_PATH is the folder that contains Code/ and Resultados/ (this copy of the pipeline).
Override with the environment variable PORTABLES_BASE. The original PORTABLES project is
never written to; its harmonized feather was copied into Resultados/ unchanged.
"""

import os

BASE_PATH = os.environ.get(
    'PORTABLES_BASE',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

if not os.path.isdir(os.path.join(BASE_PATH, 'Resultados')):
    raise RuntimeError(f"Resultados/ not found under BASE_PATH={BASE_PATH}")
