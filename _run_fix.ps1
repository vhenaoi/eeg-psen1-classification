Set-Location 'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Code'

Write-Host '============================================'
Write-Host 'PASO 1: Regenerando PSM desde HARMONIZED...'
Write-Host '============================================'
python 2_apply_psm.py 2>&1 | Out-File '_run_fix_psm.log' -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] PSM fallo'
    exit 1
}
Write-Host '[OK] PSM completado'

Write-Host '============================================'
Write-Host 'PASO 2: Ejecutando E01, E02, E04...'
Write-Host '============================================'
python run_experiments.py --ids E01 E02 E04 2>&1 | Out-File '_run_fix_experiments.log' -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] Experimentos fallaron'
    exit 1
}
Write-Host '[OK] Experimentos completados'
'DONE' | Out-File '_run_fix_COMPLETE.flag'
