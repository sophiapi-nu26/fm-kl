# Activate conda environment and run all Part-2 experiments
# This script runs experiments for all schedules and delta configurations

$condaEnvName = "flow-kl"

# Check if conda is available
try {
    conda --version | Out-Null
} catch {
    Write-Host "Error: Conda is not found. Please ensure Anaconda or Miniconda is installed and in your PATH." -ForegroundColor Red
    Exit 1
}

# Activate the conda environment
Write-Host "Activating conda environment: $condaEnvName" -ForegroundColor Green
conda activate $condaEnvName

# Check if activation was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to activate conda environment '$condaEnvName'. Please check the environment name." -ForegroundColor Red
    Exit 1
}

# Run the Python script
Write-Host "Running Part-2 automated experiments..." -ForegroundColor Green
Write-Host ""
python run_all_pt2_experiments.py

# Note: We don't check $LASTEXITCODE here because the Python script may have some failures
# but we want to see the summary of all experiments

Write-Host ""
Write-Host "Script finished." -ForegroundColor Green

