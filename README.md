# eXact thrEe-Body EigEn Solver, XEBEES or ξ3-🐝
XEBEES, pronounced ZEE-bees and also written as ξ3-🐝, is a
numerically exact eigensolver for the quantum three-body problem
developed in the Subotnik Group at Princeton University. XEBEES is
designed to benchmark our phase-space extensions to the Born
Oppenheimer approximation.

XEBEES is developed by Vale Cofer Shabica, Mansi Bhati, and Nadine
Bradbury with contributions from Alok Kumar and Xinchun Wu.

XEBEES runs on CPUs or GPUs via a variety of backends.


## Installation
Installation from source is supported via `uv` or anaconda.

### System Requirements
- **Memory**: 32GB+ recommended for large calculations
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for large calculations)

### Conda
On our cluster, Della, anaconda is already installed so we can use that to create a Python environment. First, load the modules:
```
module load anaconda3/2024.10
module load nsight-systems/2025.3.1  # optional, useful for profiling
```
Then, create the new conda environment:
```
conda env create -f environment.yml
conda activate xebees
```
**Note**: If you want to use GPU backends (like CuPy), you need to create the environment on a node with GPUs, like della-gpu.

Whenever you login, you will need to load the anaconda module and activate your saved environment. Additionally, the module load line and conda activate line will need to be included in any SLURM scripts. An example SLURM script using a GPU, `test.qs`, looks like:
```
#!/bin/bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --nodes=1           # node count
#SBATCH --ntasks=1          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G          # total memory per node
#SBATCH --time=0:10:00      # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.10
conda activate xebees

2D/fixed_center_of_mass_exact.py -k 10 -g_1 1.0 -g_2 1.0 -M_1 100 -M_2 100 -r 50 -g 50 -R 50  -t 1 --verbosity 9 --preconditioner "BO" --backend cupy &> log_test
```
You can submit the SLURM script with `sbatch test.qs`, and check the queue and get estimated job start time with `squeue -u $(USER_ID) --start`

#### Grace Hopper chip
We have experimental support for the Grace Hopper super-chip via anaconda; use `environment-gh.yml` instead of `environment.yml`.

### Using uv
First, install uv:
[https://docs.astral.sh/uv/getting-started/installation/]

If you're not worried about linking to system libraries, the installation is straightforward:
```
git clone git@github.com:subotnikgroup/xebees.git
cd xebees
uv venv
source .venv/bin/activate
uv pip install --editable .
```

Each time you want to run, you'll need to activate the venv:
```
source .venv/bin/activate
```

#### Linking to system libraries (Jupiter)
If you want to link against the system libraries, you'll need to compile some of the Python packages from source.
```
# On Jupiter, load modules
module purge
module load gcc/11.3 cmake/3.24.0 openblas/0.3.10-mp

git clone git@github.com:subotnikgroup/xebees.git
cd xebees
uv venv
source .venv/bin/activate
MAKEFLAGS="-j48" CC=$(which gcc) CXX=$(which g++) FC=$(which gfortran) uv \
  pip install --editable . --no-binary numpy --no-binary scipy --no-binary pyscf --no-binary jax
```

## Quick Start

After installation, try this simple example to verify everything works:

```bash
# Activate your environment
conda activate xebees      # for anaconda
source .venv/bin/activate  # for uv

# Run a simple 2D calculation
python 2D/fixed_center_of_mass_exact.py -g_1 1 -g_2 1 -M_1 300 -M_2 300 -k 5 -R 21 -r 22 -g 24 --verbosity 10 --preconditioner BO
```

This will:
- Use symmetric charges (`-g_1 1 -g_2 1`)
- Set equal masses (`-M_1 300 -M_2 300`)
- Calculate the lowest 5 eigenvalues (`-k 5`)
- Use a small grid size (`-R 21 -r 22 -g 24`)
- Show detailed output from the iterative solver (`--verbosity 10`)
- Use the Born-Oppenheimer preconditioner (`--preconditioner BO`)

**Expected output**: The program should complete in under a minute and display eigenvalues like so:
```
Davidson: [-0.78264809 -0.75586353 -0.73281882 -0.71379119 -0.6961425 ]
[ True  True  True  True  True]
exact gap 0.026784557996776148
All eigenvalues converged
```

## Project Structure

```
xebees/
├── 1D/     # One-dimensional calculations
│   ├── fixed_center_of_mass_exact.py   # 1D solver
│   └── notebook-1D.ipynb               # examples and tutorials
├── 2D/     # Two-dimensional calculations
│   ├── fixed_center_of_mass_exact.py   # 2D solver
│   ├── notebook-2D.ipynb               # examples and tutorials
├── lib/    # Core library modules
│   ├── xp.py                           # Backend abstraction (NumPy/CuPy/PyTorch/cuPyNumeric)
│   ├── davidson.py                     # Davidson helper functions
│   ├── hamiltonian.py                  # Kinetic energy operators
│   ├── linalg_helper.py                # Iterative eigen solver
│   └── potentials.py                   # Potential energy functions
├── tests/  # Test suite
└── environment*.yml       # Conda environment files
```

## Key Parameters

### Essential Parameters
- **`-k`**: Number of eigenvalues to compute (default: 5)
- **`-g_1`, `-g_2`**: Particle charges (required)
- **`-M_1`, `-M_2`**: Particle masses (required)
- **`-R`**: Grid points for radial R coordinate
- **`-r`**: Grid points for radial r coordinate
- **`-g`**: Grid points for angular γ coordinate
- **`-J`**: Angular momentum quantum number (default: 0)

### Advanced Parameters
- **`--backend`**: Computational backend (numpy, cupy, cupynumeric, torch)
- **`--preconditioner`**: Preconditioning method (naive, BO, jfull)
- **`--potential`**: Potential model (soft_coulomb, borgis)
- **`--verbosity`**: Output detail level (0-10)
- **`--save`**: Save resultant eigenvectors to file
- **`--guess`**: Load initial guess from file

## Testing

### Running Tests
Tests are located in the `tests` directory.

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with live output (no capture)
pytest -s

# Run specific test file
pytest tests/test_backends.py
```

### Expected Test Results
All tests should pass on a properly configured system. Common issues:
- Some backends may not be available depending on your installation
- GPU tests may fail if there is insufficient GPU memory available

## Reporting Issues
When reporting problems, include:
1. Full command line used
2. Complete error message
3. System information (OS, Python version, GPU details)
