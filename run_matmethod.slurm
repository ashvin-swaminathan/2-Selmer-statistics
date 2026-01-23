#!/bin/bash
#SBATCH --job-name=matmethod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=main
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=matmethod_%j.out
#SBATCH --error=matmethod_%j.err
#SBATCH --mail-user=ashvins@math.princeton.edu
#SBATCH --mail-type=ALL

# Print job info
echo "Job resumed on $(date)"
echo "Running on node: $SLURM_NODELIST"

cd $SLURM_SUBMIT_DIR

# Run the Python script with n=11 RESUMING from Job ID 965
# We use --resume 965 because that was the ID of the job that timed out.
python /u/ashvins/Desktop/mathmethod/matmethod_hyperfast_slurm.py 11 --resume 966

echo "Job finished on $(date)"
