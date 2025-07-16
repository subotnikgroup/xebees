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

### Using uv
First, install uv:
[https://docs.astral.sh/uv/getting-started/installation/]

If you're not worried about linking to system libraries, things are very simple:
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

Now you can run any of the included programs, e.g.:
```
1D/fixed_center_of_mass_exact.py -k 10 -g_1 1.1 -g_2 1.0 -M_1 2 -M_2 4 -r 400 -R 101 --verbosity 5
```
Or access the jupyter notebooks:
```
jupyter lab
```

#### Linking to system libraries (jupiter)
If you want to link against the system libraries, you'll need to compile some of the python packages from source.
```
# on jupiter, load modules
module purge
module load gcc/11.3 cmake/3.24.0 openblas/0.3.10-mp

git clone git@github.com:subotnikgroup/xebees.git
cd xebees
uv venv
source .venv/bin/activate
MAKEFLAGS="-j48" CC=$(which gcc) CXX=$(which g++) FC=$(which gfortran) uv \
  pip install --editable . --no-binary numpy --no-binary scipy --no-binary pyscf --no-binary jax
```

### Conda (della)
On Della, anaconda is already installed so we can use that to manage our python env's instead. First, we will need to load the modules
```
module load anaconda3/2024.10
module load nsight-systems/2025.3.1  # optional, useful for profiling
```
Then, create the new conda env
```
conda env create -f environment.yml
conda activate xebees
```

Now, you should be able to run our scripts in the head node on della.
Whenever you login, you will need to load the anaconda module and
activate your saved env. Additionally, the module load line and conda
activate line will need to be included in any slurm scripts. An
example slurm script, `test.qs` looks like:
```
#!/bin/bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --nodes=1           # node count
#SBATCH --ntasks=1          # total number of tasks across all nodes
#SBATCH --cpus-per-task=16  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G          # total memory per node
#SBATCH --time=0:10:00      # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.10
conda activate xebees

2D/fixed_center_of_mass_exact.py -k 10 -g_1 1.0 -g_2 1.0 -M_1 100 -M_2 100 -r 50 -g 50 -R 50  -t 1 --verbosity 9 --preconditioner "naive" &> log_test
```
You can submit a slurm script with `sbatch test.qs`, and check the
queue and get estimated job start time with `squeue -u $(USER_ID) --start`

#### Grace Hopper chip
We have experimental support for the Grace Hopper super-chip via anaconda; use environment-gh.yml.

## Testing
Tests are located in the `tests` directory. You can run them with `pytest`. Add the `-s` flag if you want even more output.
