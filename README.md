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
