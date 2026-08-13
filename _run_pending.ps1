& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'
conda activate base
Set-Location 'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Code'

$LogDir  = 'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados'
$TimeStr = (Get-Date -Format 'yyyyMMdd_HHmm')
$LogFile = "$LogDir\run_pending_$TimeStr.log"

function Log($msg) {
    $ts = (Get-Date -Format 'HH:mm:ss')
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content $LogFile $line
}

Log "============================================================"
Log "PIPELINE PENDIENTE: PSM fix + E08, E09, E11, E12, E13"
Log "============================================================"

# ----------------------------------------------------------------
# PASO 1: Regenerar PSM (ahora con 4:1 corregido)
# ----------------------------------------------------------------
Log "PASO 1: Regenerando PSM datasets (4:1 ahora correcto)..."
$t0 = (Get-Date)
python 2_apply_psm.py 2>&1 | Out-File "$LogDir\run_pending_psm_$TimeStr.log" -Encoding utf8
$exit_psm = $LASTEXITCODE
$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
if ($exit_psm -ne 0) {
    Log "[ERROR] PSM fallo (exit $exit_psm) en $mins min — abortando"
    exit 1
}
Log "[OK] PSM completado en $mins min"

# ----------------------------------------------------------------
# PASO 2: Ejecutar experimentos pendientes
# E08: ACr PSM 4:1 (dato corregido)
# E09: ACr PSM 5:1 (crash anterior corregido)
# E11: ACr vs HC sin residualizacion
# E12: SCr vs HC sin residualizacion
# E13: ACr vs SCr (dentro de portadores)
# ----------------------------------------------------------------
Log "PASO 2: Ejecutando E08, E09, E11, E12, E13..."
$t1 = (Get-Date)
python run_experiments.py --ids E08 E09 E11 E12 E13 2>&1 | Tee-Object -FilePath "$LogDir\run_pending_exp_$TimeStr.log" -Encoding utf8
$exit_exp = $LASTEXITCODE
$mins2 = [math]::Round(((Get-Date) - $t1).TotalMinutes, 1)
Log "[OK] Experimentos finalizados en $mins2 min (exit $exit_exp)"

# ----------------------------------------------------------------
# DONE
# ----------------------------------------------------------------
$total = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Log "============================================================"
Log "TOTAL: $total min"
Log "Logs en: $LogDir"
Log "============================================================"
'DONE' | Out-File '_run_pending_COMPLETE.flag'
