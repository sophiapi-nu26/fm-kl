# PowerShell script to run all cross-schedule evaluation experiments
# Uses trained models to evaluate KL identity across different schedules

Write-Host "=========================================="
Write-Host "Cross-Schedule KL Identity Evaluation"
Write-Host "=========================================="
Write-Host ""

$MODEL_A1 = "data/models/vtheta_schedule_a1_mse_0-05_20251027_231222.pth"
$MODEL_A2 = "data/models/vtheta_schedule_a2_mse_0-05_20251027_231335.pth"
$MODEL_A3 = "data/models/vtheta_schedule_a3_mse_0-05_20251027_231356.pth"

# Run experiments: schedule_i vs trained schedule_j where i ≠ j

Write-Host "Running (a1 vs a2 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a1 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A2"

Write-Host ""
Write-Host "Running (a1 vs a3 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a1 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A3"

Write-Host ""
Write-Host "Running (a2 vs a1 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a2 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A1"

Write-Host ""
Write-Host "Running (a2 vs a3 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a2 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A3"

Write-Host ""
Write-Host "Running (a3 vs a1 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a3 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A1"

Write-Host ""
Write-Host "Running (a3 vs a2 model) ..." -ForegroundColor Cyan
python experiment.py --schedule a3 --num_samples 2000 --num_times 101 --num_seeds 3 --load_model "$MODEL_A2"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All cross-evaluation experiments complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

