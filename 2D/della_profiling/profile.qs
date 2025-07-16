#!/bin/bash
#SBATCH --job-name=profiling     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G                 # total memory per node
#SBATCH --time=5:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.10
conda activate xebees

kernprof -l fixed_center_of_mass_exact_lineprofile.py -k 10 -g_1 1.0 -g_2 1.0 -M_1 50 -M_2 100 -r 50 -g 50 -R 50  -t 1 --verbosity 9 --preconditioner "naive" &> log_profile


