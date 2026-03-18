#!/bin/bash

Jval=(0.5 1.5)
Aval=(0e0 1e0 1e2 2e2 4e2 8e2 1e3 2e3 3e3 4e3 6e3 8e3 1e4)

for A in "${Aval[@]}"; do
    filename="J05_alpha_${A}.qs"
    cat > "$filename" << EOF
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00 
#SBATCH --constraint=gpu80

module load anaconda3/2024.10
conda activate xeebes
EOF

  for J in "${Jval[@]}"; do
  cat >> "$filename"<< EOF

python fixed_center_of_mass_exact_3D_S.py -g_1 1.0 -g_2 1.0 -M_1 2000 -M_2 2000 -R 91 -r 110 -g 50 -J ${J} -k 14 --potential erf_coulomb  --backend cupy --verbosity 5 --preconditioner naive -int 4000 --soc full --alpha ${A} --subspace 700 > log_borgis_M20_J${J}_A${A}

EOF
  done
  sbatch "$filename"
done
