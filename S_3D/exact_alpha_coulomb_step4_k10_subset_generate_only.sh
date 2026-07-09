#!/bin/bash

set -euo pipefail

# Requested exact dataset:
#   M=200, J=(1.5 2.5 6.5 7.5), alpha=(0.0 2500.0 7500.0 10000.0 12000.0)
#   M=20,  J=(1.5),             alpha=(7500.0)
RUN_SPECS=()
for ALPHA in ${ALPHA_LIST:-10000.0}; do
  for J in ${J_LIST:-8.5 9.5 10.5}; do
    RUN_SPECS+=("200|${ALPHA}|${J}")
  done
done

PYTHON_SCRIPT="S_3D/fixed_center_of_mass_exact_3D_S_full_j_scaling_step4_chunked_production.py"
PROJECT_ROOT="/home/mb3835/ps-model-exact"
SCRIPT_DIR="/home/mb3835/group_storage/step4_separate_jobs"
JOB_SCRIPT_DIR="${JOB_SCRIPT_DIR:-${SCRIPT_DIR}/exact_step4_k10_jobs}"
SUBMIT="${SUBMIT:-0}"
# auto  : each generated job looks for its own checkpoint/evecs at start-up
#         and resumes from it (via --guess) if one exists; otherwise starts fresh.
# fresh : never look for a checkpoint; always start from a random guess
#         (the old, non-resumable behavior).
RESTART_MODE="${RESTART_MODE:-auto}"

NR="${NR:-91}"
Nr="${Nr:-110}"
Ng="${Ng:-50}"
NINT="${NINT:-4000}"
KROOTS="${KROOTS:-4}"
# Per-J main Davidson subspace.  SUBSPACE still overrides all values;
# otherwise the SUBSPACE_J* knobs below are used.
SUBSPACE_OVERRIDE="${SUBSPACE:-}"
SUBSPACE_J0P5="${SUBSPACE_J0P5:-800}"
SUBSPACE_J1P5="${SUBSPACE_J1P5:-700}"
SUBSPACE_J2P5="${SUBSPACE_J2P5:-500}"
SUBSPACE_J4P5="${SUBSPACE_J4P5:-320}"
SUBSPACE_J5P5="${SUBSPACE_J5P5:-260}"
SUBSPACE_J6P5="${SUBSPACE_J6P5:-180}"
SUBSPACE_J7P5="${SUBSPACE_J7P5:-180}"
SUBSPACE_DEFAULT="${SUBSPACE_DEFAULT:-240}"
ITERATIONS="${ITERATIONS:-500}"
MATVEC_BATCH_SIZE="${MATVEC_BATCH_SIZE:-1}"
DIPOLE_G_CHUNK="${DIPOLE_G_CHUNK:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
# LOCK=1 turns on main-Davidson root locking/deflation (see
# lib/linalg_helper_locking.py / lib/linalg_helper_nvtx.py): converged roots
# stop being rebuilt from scratch on every --subspace restart. Off by
# default so existing behavior is unchanged unless explicitly requested.
LOCK="${LOCK:-0}"
LOCK_TOL_FACTOR="${LOCK_TOL_FACTOR:-10}"

DAVBO_ITERATIONS="${DAVBO_ITERATIONS:-500}"
DAVBO_SUBSPACE="${DAVBO_SUBSPACE:-300}"
DAVBO_TOL="${DAVBO_TOL:-1e-8}"
DAVBO_VERBOSITY="${DAVBO_VERBOSITY:-1}"

CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-16G}"
WALLTIME="${WALLTIME:-03:00:00}"

mkdir -p "$JOB_SCRIPT_DIR"

tag_float() {
  local value=$1
  echo "${value//./p}"
}

pick_subspace() {
  if [[ -n "$SUBSPACE_OVERRIDE" ]]; then
    echo "$SUBSPACE_OVERRIDE"
    return
  fi

  case "$1" in
    0.5) echo "$SUBSPACE_J0P5" ;;
    1.5) echo "$SUBSPACE_J1P5" ;;
    2.5) echo "$SUBSPACE_J2P5" ;;
    4.5) echo "$SUBSPACE_J4P5" ;;
    5.5) echo "$SUBSPACE_J5P5" ;;
    6.5) echo "$SUBSPACE_J6P5" ;;
    7.5) echo "$SUBSPACE_J7P5" ;;
    *) echo "$SUBSPACE_DEFAULT" ;;
  esac
}

for SPEC in "${RUN_SPECS[@]}"; do
  IFS='|' read -r MASS ALPHA J <<< "$SPEC"
  RUN_SUBSPACE=$(pick_subspace "$J")

  if [[ "$ALPHA" == "0.0" || "$ALPHA" == "0" ]]; then
    SOC_ARG="None"
  else
    SOC_ARG="full"
  fi

  JTAG=$(tag_float "$J")
  ATAG=$(tag_float "$ALPHA")
  MTAG=$(tag_float "$MASS")
  SOCTAG=${SOC_ARG,,}
  RUN_TAG="M${MTAG}_J${JTAG}_A${ATAG}_soc_${SOCTAG}"
  JOB_FILE="${JOB_SCRIPT_DIR}/exact_step4_${RUN_TAG}.slurm"

  LOCK_FLAGS=""
  if [[ "$LOCK" == "1" ]]; then
    LOCK_FLAGS="--lock --lock_tol_factor ${LOCK_TOL_FACTOR}"
  fi

  if [[ "$RESTART_MODE" == "auto" ]]; then
    # Built with a fully single-quoted heredoc so nothing expands here; all
    # $VAR references below are meant to be resolved later, at job runtime.
    GUESS_LOOKUP=$(cat <<'INNER'
GUESS_PATH=$(bash "/home/mb3835/group_storage/step4_separate_jobs/resolve_guess.sh" "$EVECS" "$CKPT_ROOT" "__RUN_TAG__")
GUESS_ARGS=()
if [[ -n "$GUESS_PATH" ]]; then
  echo "Resuming from: $GUESS_PATH"
  GUESS_ARGS=(--guess "$GUESS_PATH")
else
  echo "No existing checkpoint/evecs found; starting fresh"
fi
INNER
)
    GUESS_LOOKUP=${GUESS_LOOKUP//__RUN_TAG__/${RUN_TAG}}
  else
    GUESS_LOOKUP='GUESS_ARGS=()'
  fi

  cat > "$JOB_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=ex4_${RUN_TAG}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEMORY}
#SBATCH --gres=gpu:1
#SBATCH --time=${WALLTIME}
#SBATCH --constraint=gpu80
#SBATCH --output=${JOB_SCRIPT_DIR}/slurm-${RUN_TAG}-%j.out
#SBATCH --error=${JOB_SCRIPT_DIR}/slurm-${RUN_TAG}-%j.err

set -eo pipefail
set -o pipefail

module load anaconda3/2024.10
conda activate ps-exact

cd ${PROJECT_ROOT}

OUTPUT_ROOT="${JOB_SCRIPT_DIR}"
CKPT_ROOT="\$OUTPUT_ROOT/checkpoints"
RUN_ROOT="\$OUTPUT_ROOT/${RUN_TAG}_\${SLURM_JOB_ID}"
mkdir -p "\$RUN_ROOT" "\$CKPT_ROOT"

LOG="\$RUN_ROOT/${RUN_TAG}.log"
SMI_LOG="\$RUN_ROOT/smi_${RUN_TAG}.csv"
# Stable across resubmissions (no \${SLURM_JOB_ID}) so that a later restart of
# this same RUN_TAG finds the checkpoint/result a previous attempt left behind.
# Per-attempt artifacts (log, smi trace) stay job-ID-tagged in \$RUN_ROOT so
# restarts don't clobber earlier attempts' logs.
EVECS="\$CKPT_ROOT/${RUN_TAG}_evecs.npz"
RESULTS="\$RUN_ROOT/${RUN_TAG}_eigenvalues.txt"

echo "Job started: \$(date)"
echo "Node: \${SLURMD_NODENAME:-unknown}"
echo "Run tag: ${RUN_TAG}"
echo "Log: \$LOG"
echo "SMI: \$SMI_LOG"
echo "Evecs: \$EVECS"
echo "Results: \$RESULTS"
echo "Subspace: ${RUN_SUBSPACE}"

${GUESS_LOOKUP}

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.free \\
  --format=csv -lms 1000 > "\$SMI_LOG" &
SMI_PID=\$!

cleanup() {
  if kill -0 "\$SMI_PID" 2>/dev/null; then
    kill "\$SMI_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -u ${PYTHON_SCRIPT} \\
  -g_1 1.0 -g_2 1.0 \\
  -M_1 ${MASS} -M_2 ${MASS} \\
  -R ${NR} -r ${Nr} -g ${Ng} -int ${NINT} \\
  -J ${J} -k ${KROOTS} \\
  --potential erf_coulomb \\
  --backend cupy \\
  --verbosity 5 \\
  --preconditioner davBO \\
  --davbo_iterations ${DAVBO_ITERATIONS} \\
  --davbo_subspace ${DAVBO_SUBSPACE} \\
  --davbo_tol ${DAVBO_TOL} \\
  --davbo_verbosity ${DAVBO_VERBOSITY} \\
  --soc ${SOC_ARG} \\
  --alpha ${ALPHA} \\
  --subspace ${RUN_SUBSPACE} \\
  --iterations ${ITERATIONS} \\
  --matvec_batch_size ${MATVEC_BATCH_SIZE} \\
  --dipole_g_chunk ${DIPOLE_G_CHUNK} \\
  --checkpoint_every ${CHECKPOINT_EVERY} \\
  --evecs "\$EVECS" \\
  --save "\$RESULTS" \\
  ${LOCK_FLAGS} \\
  "\${GUESS_ARGS[@]}" \\
  2>&1 | tee "\$LOG"

PY_STATUS=\${PIPESTATUS[0]}
echo "PY_STATUS=\$PY_STATUS" | tee -a "\$LOG"
echo "Job finished: \$(date)" | tee -a "\$LOG"
exit "\$PY_STATUS"
EOF

  if [[ "$SUBMIT" == "1" ]]; then
    if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "ex4_${RUN_TAG}"; then
      echo "SKIP: ex4_${RUN_TAG} is already queued/running; not submitting" \
           "(two jobs writing the same checkpoint file would race each other)."
    else
      echo "Submitting $JOB_FILE (mass=${MASS}, alpha=${ALPHA}, J=${J}, k=${KROOTS}, subspace=${RUN_SUBSPACE}, soc=${SOC_ARG})"
      sbatch "$JOB_FILE"
    fi
  else
    echo "Generated $JOB_FILE (mass=${MASS}, alpha=${ALPHA}, J=${J}, k=${KROOTS}, subspace=${RUN_SUBSPACE}, soc=${SOC_ARG})"
  fi
done
