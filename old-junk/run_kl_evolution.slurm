#!/bin/bash
#SBATCH --job-name=kl_evolution_validation
#SBATCH --output=kl_evolution_%j.out
#SBATCH --error=kl_evolution_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=your_partition_name

# Load required modules (adjust based on your cluster)
module load python/3.11
module load cuda/11.8  # if using GPU

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fm-kl

# Set working directory
cd $SLURM_SUBMIT_DIR

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create plots directory
mkdir -p plots

# Run the experiment
echo "Starting KL evolution validation experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time: $(date)"

# Run with CPU (recommended for this experiment)
python main.py --device cpu --seed 42

echo "Experiment completed at: $(date)"
echo "Results saved in: plots/, checkpoints/, results/"
