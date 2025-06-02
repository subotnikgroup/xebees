Compute the lowest k eigenvalues of the exact Hamiltonian for a 3-body system

First, install uv:
[https://docs.astral.sh/uv/getting-started/installation/]

To set up your environment for the first time:
```
# on jupiter, load modules
module purge
module load gcc/11.3 cmake/3.24.0 openblas/0.3.10-mp 

git clone git@github.com:subotnikgroup/ps-model-exact.git
cd ps-model-exact
uv venv
source .venv/bin/activate
MAKEFLAGS="-j48" CC=$(which gcc) CXX=$(which g++) FC=$(which gfortran) uv \
  pip install --editable . --no-binary numpy --no-binary scipy --no-binary pyscf --no-binary jax
```

If you're not worried about linking to the system libraries, it's even simpler:
```
git clone git@github.com:subotnikgroup/ps-model-exact.git
cd ps-model-exact
uv venv
source .venv/bin/activate
uv pip install --editable .
```

Each time you want to run the codes, you'll activate it like so (you'll also need openblas loaded):
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

### Della conda configuration

On Della, anaconda is already installed so we can use that to manage our python env's instead. First, we will need to load the modules
```
module load anaconda/2024.10
module load nsight-systems/2025.3.1
```
Then, create the new conda env
```
conda env create -f environment.yml
conda activate ps-exact
```
Now, you should be able to run our scripts in the head node on della. Every future time you log into della, you will need to load the anaconda module and activate your saved env, the module load line and conda activate line will need to be included in any slurm scripts. An example slurm script, `test.qs` looks like:
```
#!/bin/bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=200G                 # total memory per node
#SBATCH --time=0:10:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.10
conda activate ps-exact

python 2D/fixed_center_of_mass_exact.py -k 10 -g_1 1.0 -g_2 1.0 -M_1 100 -M_2 100 -r 50 -g 50 -R 50  -t 1 --verbosity 9 --preconditioner "naive" &> log_test
```

You can submit a slurm script with `sbatch test.qs`, and check the queue and get estimated job start time with `squeue -u $(USER_ID) --start`

