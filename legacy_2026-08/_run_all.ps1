Set-Location 'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Code'
Remove-Item '_run_all_COMPLETE.flag' -ErrorAction SilentlyContinue
python run_experiments.py 2>&1 | Out-File '_run_all_experiments.log' -Encoding utf8
'DONE' | Out-File '_run_all_COMPLETE.flag'
