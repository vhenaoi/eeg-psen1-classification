"""
train_ml_v2_runner.py
=====================
Adaptador entre experiments_registry y 3_train_ml_v2.py.

Expone run_condition_from_registry(exp_config, output_dir) que traduce
un dict de experimento al API de run_condition() del pipeline principal.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import_pipeline():
    """Importa 3_train_ml_v2.py como módulo (nombre inválido para import directo)."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3_train_ml_v2.py')
    spec = importlib.util.spec_from_file_location('pipeline_v2', script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_condition_from_registry(exp_config, output_dir):
    """
    Ejecuta run_condition() con los parámetros del experiment config.

    exp_config : dict del experiments_registry
    output_dir : directorio de salida del experimento

    Retorna el dict de resultados de run_condition().
    """
    pipeline = _import_pipeline()

    condition_name   = f"{exp_config['id']}_{exp_config['name']}"
    age_strategy     = exp_config.get('age_strategy', 'residualize')
    apply_resid      = (age_strategy == 'residualize')
    include_age      = (age_strategy == 'as_feature')
    feature_family   = exp_config.get('feature_family', None)
    residualize_sex  = exp_config.get('residualize_sex', False)

    logger = pipeline.setup_logging(output_dir)

    result = pipeline.run_condition(
        condition_name=condition_name,
        filepath=exp_config['data_file'],
        output_dir=output_dir,
        logger=logger,
        apply_resid=apply_resid,
        include_age=include_age,
        exp_config=exp_config,
        feature_family=feature_family,
        residualize_sex=residualize_sex,
    )
    return result
