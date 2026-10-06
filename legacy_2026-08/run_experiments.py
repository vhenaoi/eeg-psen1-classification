"""
run_experiments.py
==================
Punto de entrada para ejecutar todos los experimentos del registry.

Uso:
    python run_experiments.py                   # corre todos
    python run_experiments.py --ids E01 E03     # corre solo E01 y E03
    python run_experiments.py --list            # muestra el registry y sale

Checkpointing:
    Si un experimento ya tiene DONE.flag en su directorio, se salta.
    Para re-ejecutar: borrar el DONE.flag correspondiente.

Orden de ejecución:
    Secuencial, en el orden del registry.
    Tiempo estimado: 20-90 min por experimento.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments_registry import get_registry, print_registry_summary
from train_ml_v2_runner    import run_condition_from_registry
from compare_experiments   import build_master_comparison

from config import BASE_PATH
EXPERIMENTS_OUT = os.path.join(BASE_PATH, 'Resultados', 'experiments')


def get_output_dir(exp):
    return os.path.join(EXPERIMENTS_OUT, f"{exp['id']}_{exp['name']}")


def run_all(exp_ids=None):
    registry = get_registry()

    if exp_ids:
        registry = [e for e in registry if e['id'] in exp_ids]
        if not registry:
            print(f"[ERROR] No se encontraron experimentos con IDs: {exp_ids}")
            return

    os.makedirs(EXPERIMENTS_OUT, exist_ok=True)

    print("\n" + "="*70)
    print("  EJECUTANDO EXPERIMENTOS")
    print("="*70)
    print(f"  Experimentos : {len(registry)}")
    print(f"  Salida       : {EXPERIMENTS_OUT}\n")

    t_global  = time.time()
    completed = []
    skipped   = []
    failed    = []

    for exp in registry:
        exp_id    = exp['id']
        out_dir   = get_output_dir(exp)
        done_flag = os.path.join(out_dir, 'DONE.flag')

        print(f"\n{'-'*70}")
        print(f"  [{exp_id}] {exp['name']}  -  {exp['comparison']}")
        print(f"  {exp['description'][:100]}")

        # Verificar archivo de datos
        if not os.path.exists(exp['data_file']):
            print(f"  [OMITIDO] Archivo no encontrado: {exp['data_file']}")
            skipped.append(exp_id)
            continue

        # Checkpointing
        if os.path.exists(done_flag):
            print(f"  [OMITIDO] Ya completado. Borrar DONE.flag para re-ejecutar.")
            skipped.append(exp_id)
            continue

        try:
            t0 = time.time()
            run_condition_from_registry(exp, out_dir)
            print(f"\n  [OK] {exp_id} completado en {(time.time()-t0)/60:.1f} min")
            completed.append(exp_id)
        except Exception as exc:
            print(f"\n  [ERROR] {exp_id} fallo: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(exp_id)

    # Resumen final
    total_min = (time.time() - t_global) / 60
    print(f"\n{'='*70}")
    print(f"  RESUMEN  ({total_min:.1f} min total)")
    print(f"  Completados : {len(completed)}  {completed}")
    print(f"  Omitidos    : {len(skipped)}    {skipped}")
    print(f"  Fallidos    : {len(failed)}     {failed}")
    print(f"{'='*70}\n")

    # Tabla maestra de comparación
    if completed or skipped:
        print("  Generando tabla comparativa maestra...")
        build_master_comparison(EXPERIMENTS_OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ejecuta experimentos del registry de ML pipeline.'
    )
    parser.add_argument(
        '--ids', nargs='+', metavar='ID',
        help='IDs a ejecutar (e.g. --ids E01 E03). Sin este flag corre todos.'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='Muestra el registry y sale sin ejecutar nada.'
    )
    args = parser.parse_args()

    if args.list:
        print_registry_summary()
        sys.exit(0)

    run_all(exp_ids=args.ids)
