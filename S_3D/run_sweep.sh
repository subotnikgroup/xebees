#!/bin/bash
masses=(2e2 20)
alphas=(0 1e2 1e3)
splits=4

# J / Ptheta / Pphi triplets (parallel arrays)
J_vals=(0.5        1.5    1.5)
Ptheta_vals=( 0.7071  1.2247 1.8708)
Pphi_vals=(0.5       1.5    0.5)

for a in "${alphas[@]}"; do
    for m in "${masses[@]}"; do
        for i in "${!J_vals[@]}"; do
            J=${J_vals[$i]}
            Pth=${Ptheta_vals[$i]}
            Pph=${Pphi_vals[$i]}

            # Sanitize decimals for filenames
            J_s=${J//./p}
            Pth_s=${Pth//./p}
            Pph_s=${Pph//./p}

            for sidx in $(seq 1 $splits); do

                filename="job_coulomb_soc_a${a}_m${m}_J${J_s}_Pth${Pth_s}_Pph${Pph_s}_sp${sidx}.qs"
                logname="log_coulomb_soc_a${a}_m${m}_J${J_s}_Pth${Pth_s}_Pph${Pph_s}_sp${sidx}"

                cat > "$filename" << EOF
#!/bin/bash
#SBATCH --job-name=soc_a${a}_m${m}_J${J}_sp${sidx}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.10
conda activate ps-exact

python S_3D/ps_cart_spin_mpi.py \\
    -g_1 1.0 -g_2 1.0 \\
    -M_1 ${m} -M_2 ${m} \\
    -x 90 -y 90 -z 90 \\
    -R 89 \\
    --potential erf_coulomb \\
    -J ${J} \\
    -Ptheta ${Pth} -Pphi ${Pph} \\
    -verbosity 5 -k 4 \\
    --backend cupy \\
    --soc full \\
    -alpha ${a} \\
    --subspace 350 \\
    -splits ${splits} -split_idx ${sidx} &> ${logname}
EOF

                echo "Generated $filename (Alpha: $a, Mass: $m, J: $J, Ptheta: $Pth, Pphi: $Pph, Split: $sidx/$splits)"
                sbatch "$filename"
                sleep 0.2

            done
        done
    done
done

