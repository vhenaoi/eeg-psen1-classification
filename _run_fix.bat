@echo off
cd /d "E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Code"
echo ============================================
echo PASO 1: Regenerando PSM desde HARMONIZED...
echo ============================================
python 2_apply_psm.py > _run_fix_psm.log 2>&1
if errorlevel 1 (
    echo [ERROR] PSM fallo. Ver _run_fix_psm.log
    exit /b 1
)
echo [OK] PSM completado.

echo ============================================
echo PASO 2: Ejecutando E01, E02, E04...
echo ============================================
python run_experiments.py --ids E01 E02 E04 > _run_fix_experiments.log 2>&1
if errorlevel 1 (
    echo [ERROR] Experimentos fallaron. Ver _run_fix_experiments.log
    exit /b 1
)
echo [OK] Experimentos completados.
echo DONE > _run_fix_COMPLETE.flag
