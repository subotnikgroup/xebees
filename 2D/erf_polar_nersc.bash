#!/bin/bash

values=(2 3 5 6 7 8 9 10 15 20 30 50 70 90)

NR=90
NCPU=128

for M in "${values[@]}"; do
  filename="job_M${M}_polar_erf_j10.qs"
  cat > "$filename" << EOF
#!/bin/bash
#SBATCH -C cpu
#SBATCH --job-name=ex${M}_polar_erf_j10 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=$NCPU
#SBATCH --account=m3138
#SBATCH -q regular

module load python/3.13 conda
conda activate xebees


python mem_polar_cpu_erf.py -g_1 0.5 -g_2 1.0 -M_1 1e6 -M_2 ${M} -r $NR -g $NR -R $NR --potential borgis -J 10 -t $NR --evecs ${M}_polar_erf_j10.npz &> log_${M}_polar_erf_j10
EOF

  echo "Generated $filename"
  echo "Submitting $filename"
  sbatch "$filename"
done
