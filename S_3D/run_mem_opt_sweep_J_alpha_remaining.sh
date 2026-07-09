#!/bin/bash

set -euo pipefail

SWEEP_ALPHAS=(0.0 1000.0 2500.0 7500.0 10000.0 12000.0)

# These values are coupled by array index.
BASE_SWEEP_J=(0 0.5 1 1.5 2 2.5 3 7 7.5 8 9 9.5 10)
BASE_SWEEP_PPHI=(0 0.5 1 1.5 2 2.5 3 7 7.5 8 9 9.5 10)
BASE_SWEEP_PTHETA=(0.0 0.707 1.0 1.224 1.414 1.58 1.73 2.64 2.73 2.83 3.0 3.08 3.16)

SWEEP_MASSES=(20 200 2000)
SPLITS=6
POLISH_GUESS_ARGS="--phase4-rbm-svd-tol 1e-10 --phase4-rbm-bank-size 4 --phase4-rbm-store-size 8 --phase4-rbm-polish-guess"

if (( ${#BASE_SWEEP_J[@]} != ${#BASE_SWEEP_PTHETA[@]} ||
      ${#BASE_SWEEP_J[@]} != ${#BASE_SWEEP_PPHI[@]} )); then
    echo "BASE_SWEEP_J, BASE_SWEEP_PTHETA, and BASE_SWEEP_PPHI must have equal lengths." >&2
    exit 1
fi

tag_float() {
    local value=$1
    echo "${value//./p}"
}

spin_mode_for_J() {
    local J=$1
    if [[ "$J" =~ ^-?[0-9]+\.5$ ]]; then
        echo "spin"
    else
        echo "no_spin_terms"
    fi
}

already_done() {
    local mass=$1
    local J=$2

    if [[ "$mass" != "2000" ]]; then
        return 1
    fi

    case "$J" in
        0|0.5|1|1.5|2)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

for mass in "${SWEEP_MASSES[@]}"; do
    for alpha in "${SWEEP_ALPHAS[@]}"; do
        if [[ "$alpha" == "0.0" || "$alpha" == "0" ]]; then
            SOC_ARG="no_soc"
        else
            SOC_ARG="full"
        fi

        for i in "${!BASE_SWEEP_J[@]}"; do
            J=${BASE_SWEEP_J[$i]}
            Ptheta=${BASE_SWEEP_PTHETA[$i]}
            Pphi=${BASE_SWEEP_PPHI[$i]}

            if already_done "$mass" "$J"; then
                echo "Skipping completed calculation (M=${mass}, alpha=${alpha}, J=${J})"
                continue
            fi

            spin_mode=$(spin_mode_for_J "$J")
            spin_args=""
            if [[ "$spin_mode" == "no_spin_terms" ]]; then
                spin_args="--no_spin_terms"
            fi

            mass_tag=$(tag_float "$mass")
            alpha_tag=$(tag_float "$alpha")
            J_tag=$(tag_float "$J")
            Ptheta_tag=$(tag_float "$Ptheta")
            Pphi_tag=$(tag_float "$Pphi")
            soc_tag=${SOC_ARG//_/-}
            parameter_tag="M${mass_tag}_a${alpha_tag}_J${J_tag}_Pth${Ptheta_tag}_Pph${Pphi_tag}_soc${soc_tag}"

            for split_idx in $(seq 1 "$SPLITS"); do
                job_file="mem_opt_${spin_mode}_${parameter_tag}_split${split_idx}.qs"
                log_file="mem_opt_${spin_mode}_${parameter_tag}_split${split_idx}.log"

                cat > "$job_file" << EOF
#!/bin/bash
#SBATCH --job-name=mem_${spin_mode}_${parameter_tag}_s${split_idx}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2024.10
conda activate ps-exact

python S_3D/mem_opt_rbm_streamed.py \\
    -g_1 1.0 -g_2 1.0 \\
    -M_1 ${mass} -M_2 ${mass} \\
    -x 90 -y 90 -z 90 \\
    -R 89 \\
    --potential erf_coulomb \\
    -Ptheta ${Ptheta} -Pphi ${Pphi} \\
    --verbosity 5 -k 4 \\
    --backend cupy \\
    --soc ${SOC_ARG} \\
    -alpha ${alpha} \\
    --iterations 5000 \\
    --subspace 125 \\
    -J ${J} \\
    -splits ${SPLITS} -split_idx ${split_idx} \\
    --Gammasq ${spin_args} ${POLISH_GUESS_ARGS} &> ${log_file}
EOF

                echo "Generated ${job_file} (mode=${spin_mode}, M=${mass}, alpha=${alpha}, J=${J}, Ptheta=${Ptheta}, Pphi=${Pphi}, soc=${SOC_ARG}, split=${split_idx}/${SPLITS}, rbm=polish_guess)"
                sbatch "$job_file"
                sleep 0.1
            done
        done
    done
done
