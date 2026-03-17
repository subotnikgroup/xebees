#!/bin/bash


Mval=(3 4 6 8 10 15 20 30 40 60 80 100 140 180 220 300)

for M in "${Mval[@]}"; do
    filename="ps_m${M}.qs"
    cat > "$filename" << EOF
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2 
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00 
#SBATCH --constraint=gpu80

module load anaconda3/2024.10
conda activate xeebes

python ps_cart_3D_fix_erf.py -g_1 0.5 -g_2 1.0 -M_1 1e6 -M_2 ${M} -x 91 -y 91 -z 91 -R 91 --potential borgis -Ptheta 0 -Pphi 0 -t 4  --verbosity 0 -k 2 --backend cupy --evecs PS_M${M}_J0.npz > log_borgis_PS_${M}_J0
#python ps_cart_3D_fix_erf.py -g_1 0.5 -g_2 1.0 -M_1 1e6 -M_2 ${M} -x 91 -y 91 -z 91 -R 91 --potential borgis -Ptheta 1 -Pphi 1 -t 4  --verbosity 0 -k 2 --backend cupy --evecs PS_M${M}_J1.npz > log_borgis_PS_${M}_J1
#python ps_cart_3D_fix_erf.py -g_1 0.5 -g_2 1.0 -M_1 1e6 -M_2 ${M} -x 91 -y 91 -z 91 -R 91 --potential borgis -Ptheta 2 -Pphi 1.4142 -t 4  --verbosity 0 -k 2 --backend cupy --evecs PS_M${M}_J2.npz > log_borgis_PS_${M}_J2

EOF
  sbatch "$filename"
done
