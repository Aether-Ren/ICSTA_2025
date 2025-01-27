#!/bin/bash
#SBATCH --partition=maths
#SBATCH --job-name=python_array_jobs
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-user=2633042r@glasgow.ac.uk
#SBATCH --mail-type=ALL

module load anaconda
source activate GT


SCRIPT_DIR="/ToyProblem"


SCRIPT_FILE=$(ls $SCRIPT_DIR/*.py | sed -n "${SLURM_ARRAY_TASK_ID}p")


echo "Running $SCRIPT_FILE on SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
python $SCRIPT_FILE
