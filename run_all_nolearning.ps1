# PowerShell script to run all no-learning test permutations
# Usage: .\run_all_nolearning.ps1

Write-Host "=========================================="
Write-Host "Running All No-Learning Test Permutations"
Write-Host "=========================================="
Write-Host ""

# Run all permutations: (a1,a2), (a1,a3), (a2,a1), (a2,a3), (a3,a1), (a3,a2)

Write-Host "Running (a1, a2)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a1 --schedule_q a2 --skip_ode

Write-Host ""
Write-Host "Running (a1, a3)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a1 --schedule_q a3 --skip_ode

Write-Host ""
Write-Host "Running (a2, a1)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a2 --schedule_q a1 --skip_ode

Write-Host ""
Write-Host "Running (a2, a3)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a2 --schedule_q a3 --skip_ode

Write-Host ""
Write-Host "Running (a3, a1)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a3 --schedule_q a1 --skip_ode

Write-Host ""
Write-Host "Running (a3, a2)..." -ForegroundColor Cyan
python nolearning_test.py --schedule_p a3 --schedule_q a2 --skip_ode

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "All no-learning tests complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

