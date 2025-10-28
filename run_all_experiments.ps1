# PowerShell script to run all experiments in conda environment
# Usage: .\run_all_experiments.ps1

Write-Host "`n======================================"
Write-Host "Activating conda environment: flow-kl"
Write-Host "======================================`n"

# Activate conda environment
conda activate flow-kl

Write-Host "Environment activated.`n"

# Run the Python script
python run_all_experiments.py

Write-Host "`n======================================"
Write-Host "All experiments complete!"
Write-Host "======================================`n"

