"""
experiments_registry.py
========================
Fuente de verdad de TODOS los experimentos del pipeline.

Cada experimento es un dict con:
  id                   : identificador único (e.g. 'E01')
  name                 : nombre corto para carpeta y tablas
  stage                : 1=población genética E280A, 2=generalización esporádicos
  comparison           : etiqueta de la comparación (e.g. 'ACr_vs_HC')
  case_label           : etiqueta del grupo caso en el feather (grupo mapeado)
  control_label        : etiqueta del grupo control en el feather
  case_orig_labels     : lista de orig_group a incluir como casos
                         (None = usar todos los que tengan case_label)
  control_orig_labels  : lista de orig_group a incluir como controles
                         (None = usar todos los que tengan control_label)
  data_file            : ruta absoluta al feather de entrada
  balancing            : 'psm_1to1' | 'psm_2to1' | 'psm_5to1' | 'none'
  smote                : True/False  (SMOTE dentro de folds de CV)
  age_strategy         : 'residualize' | 'as_feature' | 'none'
  description          : texto libre para documentar el experimento

Nota sobre class_weight:
  Todos los clasificadores (RF, SVM, LR) tienen class_weight='balanced'
  en sus grids de hiperparámetros → siempre activo. La columna 'balancing'
  refleja si hay PSM o SMOTE adicional; class_weight es implícita.

Cómo agregar un experimento nuevo:
  1. Añadir un dict al return de get_registry()
  2. Incrementar el ID (E08, E09, …)
  3. El archivo data_file debe existir antes de ejecutar run_experiments.py
"""

import os

from config import BASE_PATH


def _results(*parts):
    return os.path.join(BASE_PATH, 'Resultados', *parts)


def _psm(*parts):
    return os.path.join(BASE_PATH, 'Resultados', 'PSM_datasets', *parts)


def _experiments_out(*parts):
    return os.path.join(BASE_PATH, 'Resultados', 'experiments', *parts)


def get_data_file():
    """Retorna el archivo base de datos (harmonizado si existe, sino raw)."""
    harmonized = _results('Data_complete_ce_roi_HARMONIZED.feather')
    baseline   = _results('Data_complete_ce_roi.feather')
    return harmonized if os.path.exists(harmonized) else baseline


def get_registry():
    """
    Retorna lista de configs de experimentos.
    Los experimentos cuyo data_file no existe se pueden ejecutar igual
    (run_experiments.py los saltará con advertencia).
    """
    base_data = get_data_file()

    registry = [

        # =====================================================================
        # STAGE 1 — ACr (portadores asintomáticos E280A) vs Controles
        # =====================================================================

        {
            'id':                   'E01',
            'name':                 'ACr_PSM1to1_resid',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,   # todos los PSEN1 del archivo PSM
            'control_orig_labels':  None,
            'data_file':            _psm('Data_matched_ce_roi_PSEN1_1to1.feather'),
            'balancing':            'psm_1to1',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs HC pareados 1:1 por edad (PSM primario). '
                'Condicion principal: maxima comparabilidad demografica.'
            ),
        },

        {
            'id':                   'E02',
            'name':                 'ACr_PSM2to1_resid',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            _psm('Data_matched_ce_roi_PSEN1_2to1.feather'),
            'balancing':            'psm_2to1',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs HC pareados 2:1 por edad (PSM primario). '
                'Mayor N de controles que E01, menor desbalance.'
            ),
        },

        {
            'id':                   'E03',
            'name':                 'ACr_noPSM_classweight',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,   # todos los ACr/GG/G1
            'control_orig_labels':  None,   # todos los controles (683)
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs TODOS los controles disponibles. Sin PSM, sin SMOTE. '
                'Manejo del desbalance solo via class_weight=balanced (siempre en grids). '
                'Permite usar el maximo de datos disponibles.'
            ),
        },

        {
            'id':                   'E04',
            'name':                 'ACr_noPSM_smote',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs TODOS los controles. Sin PSM, con SMOTE dentro de folds. '
                'Doble estrategia de balanceo: SMOTE + class_weight.'
            ),
        },

        # =====================================================================
        # STAGE 1 — SCr (portadores sintomáticos E280A) vs Controles HC_SCr
        #
        # REQUISITO: orig_group debe existir en el feather (generado por
        # 1_make_dataframe.py ya actualizado). Para E05 con PSM, se necesita
        # ejecutar 2_apply_psm.py específicamente para SCr (pendiente).
        # =====================================================================

        {
            'id':                   'E05',
            'name':                 'SCr_noPSM_smote',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',          # label interno para este exp
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],         # filtra orig_group == 'SCr'
            'control_orig_labels':  None,            # todos los controles
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'SCr (portadores sintomaticos) vs todos los controles. '
                'Sin PSM. Requiere columna orig_group en el feather. '
                'Referencia: 58 SCr vs ~131 HC_SCr potencialmente pareables.'
            ),
        },

        {
            'id':                   'E06',
            'name':                 'SCr_noPSM_classweight',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'description': (
                'SCr vs todos los controles. Sin PSM, sin SMOTE. '
                'Solo class_weight=balanced. Contraste con E05.'
            ),
        },

        # =====================================================================
        # STAGE 1 — ACr solo vs HC_ACr (controles específicos, sin mezclar)
        # Usa orig_group para filtrar solo HC_ACr del pool de controles.
        # =====================================================================

        {
            'id':                   'E07',
            'name':                 'ACr_HCspecific_noPSM',
            'stage':                1,
            'comparison':           'ACr_vs_HC_ACr_only',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  ['HC_ACr', 'CTR'],  # solo controles específicos ACr
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs HC_ACr únicamente (sin mezclar HC_SCr/HC_AD/otros). '
                'Sin PSM. Permite ver el efecto de usar controles específicos '
                'vs el pool completo (comparar con E03/E04).'
            ),
        },

        # =====================================================================
        # STAGE 1 — ACr vs HC, PSM ratios mayores (4:1 y 5:1)
        # Exploran el trade-off entre comparabilidad demográfica (PSM) y
        # tamaño muestral. Datasets generados desde HARMONIZED (May 7 2026).
        # =====================================================================

        {
            'id':                   'E08',
            'name':                 'ACr_PSM4to1_resid',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            _psm('Data_matched_ce_roi_PSEN1_4to1.feather'),
            'balancing':            'psm_4to1',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs HC pareados 4:1 por edad (PSM primario, harmonizado). '
                'Mayor N que E02 (2:1), manteniendo age-matching. '
                'Explora si más controles PSM mejora AUC respecto a E02.'
            ),
        },

        {
            'id':                   'E09',
            'name':                 'ACr_PSM5to1_resid',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            _psm('Data_matched_ce_roi_PSEN1_5to1.feather'),
            'balancing':            'psm_5to1',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'ACr vs HC pareados 5:1 por edad (PSM primario, harmonizado). '
                'Mayor ratio PSM disponible. Límite superior del age-matching '
                'antes de pasar a sin-PSM (E03). Comparar con E01/E02/E08/E03.'
            ),
        },

        # =====================================================================
        # STAGE 1 — SCr vs HC con XGBoost (E10)
        # Replica E05 con clasificador adicional XGBoost. Objetivo: AUC > 0.90.
        # XGBoost agrega XGB_kbest y XGB_rfe como opciones; ENS_soft también
        # incorpora XGB. Comparar AUC vs E05 (LR_rfe=0.871).
        # =====================================================================

        {
            'id':                   'E10',
            'name':                 'SCr_noPSM_smote_xgb',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'SCr vs todos los controles. Mismo que E05 pero con XGBoost '
                'añadido al grid (XGB_kbest, XGB_rfe, ENS_soft actualizado). '
                'Objetivo: superar AUC=0.871 (E05 LR_rfe) hacia 0.90.'
            ),
        },

        # =====================================================================
        # STAGE 1 — Ablación: sin residualización de edad, sin edad como feature
        # Compara directamente con E03 (ACr) y E05 (SCr) para cuantificar
        # cuánto del AUC se debe a la corrección de edad vs. señal EEG pura.
        # age_strategy='none' → edad ignorada completamente.
        # =====================================================================

        {
            'id':                   'E11',
            'name':                 'ACr_noPSM_cw_no_age',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'none',
            'description': (
                'ACr vs HC. Sin PSM, sin SMOTE. Igual que E03 EXCEPTO: '
                'edad ignorada completamente (sin residualizar, sin incluir). '
                'Ablacion pura: cuantifica cuanto AUC viene de señal EEG vs. correccion de edad.'
            ),
        },

        {
            'id':                   'E12',
            'name':                 'SCr_noPSM_smote_no_age',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'none',
            'description': (
                'SCr vs HC. Sin PSM, con SMOTE. Igual que E05 EXCEPTO: '
                'edad ignorada completamente (sin residualizar, sin incluir). '
                'Ablacion pura: cuantifica cuanto AUC viene de señal EEG vs. correccion de edad.'
            ),
        },

        # =====================================================================
        # STAGE 1 — Clasificación DENTRO de portadores PSEN1 E280A
        # Pregunta del director: ¿puede el EEG distinguir ACr de SCr?
        # Sin controles sanos: misma familia, mismo gen, mismo sitio (Medellín).
        # Si funciona → EEG detecta la conversión a sintomático independiente
        # de diferencias poblacionales. Prueba más exigente y más limpia.
        # =====================================================================

        {
            'id':                   'E13',
            'name':                 'ACr_vs_SCr_intracarriers',
            'stage':                1,
            'comparison':           'ACr_vs_SCr',
            'case_label':           'SCr',
            'control_label':        'PSEN1',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'SCr (sintomaticos) vs ACr (asintomaticos) — clasificacion '
                'dentro de portadores PSEN1 E280A unicamente. '
                'Misma familia, mismo sitio: sin confound poblacional ni de sitio. '
                'Pregunta: detecta EEG la conversion a sintomatico? '
                'Clase positiva=SCr (y=1), negativa=ACr/PSEN1 (y=0). '
                'N aprox: 47 SCr vs 91 ACr; SMOTE para balanceo.'
            ),
        },

        # =====================================================================
        # STAGE 1 — Validación intra-sitio: solo SITE='Medellin_ld'
        # Sin confound de sitio en absoluto (mismo equipo y protocolo).
        # Valida que las diferencias EEG no son artefacto cross-site
        # y que la harmonización es correcta.
        # =====================================================================

        {
            'id':                   'E14',
            'name':                 'ACr_vs_HC_Medellin_ld',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'site_filter':          'Medellin_ld',
            'description': (
                'ACr vs HC restringido a SITE=Medellin_ld. '
                'Sin confound de sitio: mismos equipos y protocolo. '
                'Valida que AUC no depende de diferencias cross-site. '
                'Comparar con E03 (multisite): si AUC similar -> harmonizacion correcta.'
            ),
        },

        # =====================================================================
        # STAGE 1 — Ablación por familia de features
        # Responde: ¿qué tipo de medida EEG aporta la discriminación?
        # Cada experimento usa la misma config que E03 (ACr) o E05 (SCr),
        # variando únicamente el subconjunto de features:
        #   power (64)     : potencia espectral absoluta {ROI}_{Banda}
        #   sl (64)        : Synchronization Likelihood {ROI}_{Banda}_sl
        #   coh (64)       : Coherencia por banda {ROI}_{Banda}_coh
        #   entropy (64)   : Entropía {ROI}_{Banda}_ent
        #   crossfreq (288): Ratios inter-banda {ROI}_{Banda}/M{Banda}
        #   node_level (256): power + sl + coh + entropy (todo sin ratios)
        # =====================================================================

        # --- ACr vs HC (base: E03 config — sin PSM, class_weight, resid edad) ---

        {
            'id':                   'E15',
            'name':                 'ACr_noPSM_cw_power_only',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'power',
            'description': (
                'ACr vs HC. Config E03 + solo potencia espectral absoluta (64 features). '
                'Ablacion: cuantifica cuanto AUC viene de power vs otras medidas.'
            ),
        },

        {
            'id':                   'E16',
            'name':                 'ACr_noPSM_cw_sl_only',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'sl',
            'description': (
                'ACr vs HC. Config E03 + solo Synchronization Likelihood (64 features). '
                'Ablacion: cuantifica cuanto AUC viene de conectividad SL.'
            ),
        },

        {
            'id':                   'E17',
            'name':                 'ACr_noPSM_cw_coh_only',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'coh',
            'description': (
                'ACr vs HC. Config E03 + solo coherencia por banda (64 features). '
                'Ablacion: cuantifica cuanto AUC viene de coherencia inter-ROI.'
            ),
        },

        {
            'id':                   'E18',
            'name':                 'ACr_noPSM_cw_entropy_only',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'entropy',
            'description': (
                'ACr vs HC. Config E03 + solo entropia por banda (64 features). '
                'Ablacion: cuantifica cuanto AUC viene de complejidad de senal.'
            ),
        },

        {
            'id':                   'E19',
            'name':                 'ACr_noPSM_cw_crossfreq_only',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'crossfreq',
            'description': (
                'ACr vs HC. Config E03 + solo ratios inter-banda (288 features). '
                'Ablacion: cuantifica cuanto AUC viene de relaciones entre bandas.'
            ),
        },

        {
            'id':                   'E20',
            'name':                 'ACr_noPSM_cw_node_level',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'feature_family':       'node_level',
            'description': (
                'ACr vs HC. Config E03 + power+SL+coh+entropy sin ratios (256 features). '
                'Ablacion: node-level vs cross-frequency (E19): cual grupo aporta mas?'
            ),
        },

        # --- SCr vs HC (base: E05 config — sin PSM, SMOTE, resid edad) ---

        {
            'id':                   'E21',
            'name':                 'SCr_noPSM_smote_power_only',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'power',
            'description': (
                'SCr vs HC. Config E05 + solo potencia espectral absoluta (64 features).'
            ),
        },

        {
            'id':                   'E22',
            'name':                 'SCr_noPSM_smote_sl_only',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'sl',
            'description': (
                'SCr vs HC. Config E05 + solo Synchronization Likelihood (64 features).'
            ),
        },

        {
            'id':                   'E23',
            'name':                 'SCr_noPSM_smote_coh_only',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'coh',
            'description': (
                'SCr vs HC. Config E05 + solo coherencia por banda (64 features).'
            ),
        },

        {
            'id':                   'E24',
            'name':                 'SCr_noPSM_smote_entropy_only',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'entropy',
            'description': (
                'SCr vs HC. Config E05 + solo entropia por banda (64 features).'
            ),
        },

        {
            'id':                   'E25',
            'name':                 'SCr_noPSM_smote_crossfreq_only',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'crossfreq',
            'description': (
                'SCr vs HC. Config E05 + solo ratios inter-banda (288 features).'
            ),
        },

        {
            'id':                   'E26',
            'name':                 'SCr_noPSM_smote_node_level',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'feature_family':       'node_level',
            'description': (
                'SCr vs HC. Config E05 + power+SL+coh+entropy sin ratios (256 features).'
            ),
        },

        # =====================================================================
        # STAGE 1 — Sensibilidad a confusor de sexo
        # Residualiza edad Y sexo dentro de cada fold.
        # Responde: ¿el AUC se mantiene al eliminar el efecto del sexo?
        # Compara con E03 (ACr) y E05 (SCr) donde solo se residualiza edad.
        # =====================================================================

        {
            'id':                   'E27',
            'name':                 'ACr_noPSM_cw_resid_age_sex',
            'stage':                1,
            'comparison':           'ACr_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'residualize_sex':      True,
            'description': (
                'ACr vs HC. Config E03 + residualizacion de edad Y sexo dentro de folds. '
                'Verifica que AUC no depende de diferencias de sexo entre grupos. '
                'Si AUC similar a E03 -> sexo no es confusor; si baja mucho -> hay confusor.'
            ),
        },

        {
            'id':                   'E28',
            'name':                 'SCr_noPSM_smote_resid_age_sex',
            'stage':                1,
            'comparison':           'SCr_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'residualize_sex':      True,
            'description': (
                'SCr vs HC. Config E05 + residualizacion de edad Y sexo dentro de folds. '
                'Verifica que AUC no depende de diferencias de sexo entre grupos.'
            ),
        },

        # =====================================================================
        # STAGE 1 — E29: Todos los portadores PSEN1 E280A (ACr+SCr) vs HC
        # Pregunta clínica: ¿puede el EEG detectar el estado de portador
        # independientemente del estadio clínico (asintomático o sintomático)?
        # Combina ACr (orig_group: 'ACr','GG','G1') + SCr (orig_group: 'SCr')
        # → 138 casos vs 683 controles (ratio 1:5, mucho mejor que E05 1:14.5).
        # Mayor poder estadístico y F1/precision más interpretables.
        # =====================================================================

        {
            'id':                   'E29',
            'name':                 'AllCarriers_vs_HC_smote',
            'stage':                1,
            'comparison':           'AllPSEN1_vs_HC',
            'case_label':           'AllCarriers',
            'control_label':        'Control',
            'case_orig_labels':     ['ACr', 'GG', 'G1', 'SCr'],  # todos los portadores E280A
            'control_orig_labels':  None,                          # todos los 683 controles
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'residualize_sex':      True,
            'description': (
                'Todos los portadores PSEN1 E280A (ACr+SCr, ~138 casos) vs todos '
                'los controles HC (~683). Ratio 1:5 vs 1:14.5 de E05 → mejor precision/F1. '
                'Pregunta: ¿detecta EEG el estado de portador independientemente del estadio? '
                'Residualiza edad+sexo. Comparar AUC con E03 (ACr solo) y E05 (SCr solo).'
            ),
        },

        # =====================================================================
        # STAGE 1 — E30: SCr vs HC_SCr (site-matched, test más duro)
        # Usa SOLO controles del mismo sitio que los SCr (Medellín_ld).
        # HC_SCr (73 sujetos) son controles sanos del mismo protocolo que SCr.
        # Elimina completamente el confound de sitio que podría inflar E05.
        # Si AUC ≥ 0.75 → el resultado E05 no es artefacto de sitio.
        # Equivalente metodológico de E07 (ACr vs HC_ACr) pero para SCr.
        # =====================================================================

        {
            'id':                   'E30',
            'name':                 'SCr_vs_HCSCr_sitematch',
            'stage':                1,
            'comparison':           'SCr_vs_HC_SCr',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  ['HC_SCr'],   # solo controles del mismo sitio
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'description': (
                'SCr (portadores sintomáticos, ~47) vs HC_SCr (controles Medellín, ~73). '
                'N=120, mismo sitio, mismo protocolo: cero confound de sitio. '
                'Test más exigente de E05: si AUC se mantiene ≥0.75 -> '
                'el efecto es real, no artefacto cross-site. '
                'Equivalente de E07 (ACr vs HC_ACr) para portadores sintomáticos.'
            ),
        },

        # =====================================================================
        # STAGE 1 — Impacto del equipo de grabación en clasificación
        #
        # Motivación: hay 3 sites de Medellín:
        #   Medellín_hd  → alta resolución (equipo principal)
        #   Medellin_ld  → baja resolución (equipo portátil)
        #   Medellin_duque → alta resolución, época diferente (equipo incierto)
        #
        # E31/E32 excluyen portadores (casos) del equipo ld usando el nuevo
        # parámetro case_site_exclude, manteniendo todos los controles.
        # Comparar AUC con E29 (E31) y E05 (E32) revela si el equipo ld
        # degrada la clasificación o si el efecto es robusto al equipo.
        # =====================================================================

        {
            'id':                   'E31',
            'name':                 'AllCarriers_noLD_vs_HC',
            'stage':                1,
            'comparison':           'AllPSEN1_noLD_vs_HC',
            'case_label':           'AllCarriers',
            'control_label':        'Control',
            'case_orig_labels':     ['ACr', 'GG', 'G1', 'SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'residualize_sex':      True,
            'case_site_exclude':    ['Medellin_ld'],
            'description': (
                'Todos los portadores PSEN1 E280A (ACr+SCr) EXCEPTO los de Medellin_ld '
                '(equipo baja resolución) vs todos los controles HC. '
                'Replica E29 sin portadores del equipo portátil. '
                'Si AUC > E29: el equipo ld degrada la clasificación. '
                'Si AUC ≈ E29: el efecto de portador es robusto al equipo.'
            ),
        },

        {
            'id':                   'E32',
            'name':                 'SCr_noLD_vs_HC',
            'stage':                1,
            'comparison':           'SCr_noLD_vs_HC',
            'case_label':           'SCr',
            'control_label':        'Control',
            'case_orig_labels':     ['SCr'],
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                True,
            'age_strategy':         'residualize',
            'case_site_exclude':    ['Medellin_ld'],
            'description': (
                'SCr (portadores sintomáticos) EXCEPTO los de Medellin_ld vs todos los HC. '
                'Replica E05 sin portadores del equipo portátil. '
                'E05 es el resultado principal (AUC 0.871): si E32 >> E05, '
                'los SCr del equipo ld están bajando el AUC del resultado principal. '
                'Si E32 ≈ E05: los SCr ld no son un subgrupo problemático.'
            ),
        },

        # =====================================================================
        # STAGE 1 — E33: ACr sin equipo portátil vs HC (espejo de E32 para ACr)
        #
        # Motivación: E32 mostró el impacto del equipo ld en SCr (boot 0.788 vs
        # E05 0.836). Para ACr el impacto potencial es mayor: 45% de los ACr
        # son de Medellin_ld (41/91) vs 34% de los SCr (16/47).
        # Replica E03 (referencia ACr) con case_site_exclude=['Medellin_ld'].
        # N resultante: ~50 ACr (HD+Duque) vs 683 HC.
        # =====================================================================

        {
            'id':                   'E33',
            'name':                 'ACr_noLD_vs_HC',
            'stage':                1,
            'comparison':           'ACr_noLD_vs_HC',
            'case_label':           'PSEN1',
            'control_label':        'Control',
            'case_orig_labels':     None,
            'control_orig_labels':  None,
            'data_file':            base_data,
            'balancing':            'none',
            'smote':                False,
            'age_strategy':         'residualize',
            'case_site_exclude':    ['Medellin_ld'],
            'description': (
                'ACr (portadores asintomáticos) EXCEPTO los de Medellin_ld vs todos los HC. '
                'Replica E03 sin portadores del equipo portátil (N: ~50 ACr vs 683 HC). '
                'Simétrico a E32 (SCr_noLD_vs_HC) para completar el análisis de equipo. '
                'Si AUC ≈ E03 (0.783): equipo ld no afecta ACr. '
                'Si AUC >> E03: el 45% de ACr en ld estaba degradando el resultado.'
            ),
        },

        # =====================================================================
        # STAGE 2 — Generalización a poblaciones esporádicas (pendiente)
        # (AD esporádico y MCI de Grecia y Seoul)
        # Ejecutar DESPUÉS de tener final_model.pkl de E03/E05.
        # =====================================================================

    ]

    return registry


def get_experiment_by_id(exp_id):
    """Retorna la config de un experimento por ID. Lanza KeyError si no existe."""
    for exp in get_registry():
        if exp['id'] == exp_id:
            return exp
    raise KeyError(f"Experimento '{exp_id}' no encontrado en el registry.")


def print_registry_summary():
    """Imprime tabla resumen de todos los experimentos definidos."""
    registry = get_registry()
    header = f"{'ID':<5} {'Nombre':<28} {'Comparación':<22} {'Balanceo':<12} {'SMOTE':<6} {'Age':<12} {'Archivo'}"
    print("\n" + "="*110)
    print("  REGISTRO DE EXPERIMENTOS")
    print("="*110)
    print(f"  {header}")
    print("  " + "-"*107)
    for exp in registry:
        fname = os.path.basename(exp['data_file'])
        exists = "[OK]" if os.path.exists(exp['data_file']) else "[?]"
        print(f"  {exp['id']:<5} {exp['name']:<28} {exp['comparison']:<22} "
              f"{exp['balancing']:<12} {'Si' if exp['smote'] else 'No':<6} "
              f"{exp['age_strategy']:<12} {exists} {fname}")
    print("="*110)
    print(f"  Total: {len(registry)} experimentos definidos")
    print()


if __name__ == '__main__':
    print_registry_summary()
