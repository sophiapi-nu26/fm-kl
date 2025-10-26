# Setup script for KL Evolution Identity Validation

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate fm-kl

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchdiffeq; print('torchdiffeq installed successfully')"

echo "Environment setup complete!"
echo "To activate: conda activate fm-kl"
echo "To run experiment: python main.py"
