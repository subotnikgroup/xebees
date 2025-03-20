Compute the lowest k eigenvalues of the exact Hamiltonian for a 3-body system

To set up your environment do the following:
```
uv venv
uv pip install --editable .
```
Now, you can activate it like so (every time):
```
source .venv/bin/activate
```
And run any of the included programs, e.g.:
```
1D/fixed_center_of_mass_exact.py -k 10 -g_1 1.1 -g_2 1.0 -M_1 2 -M_2 4 -r 400 -R 101 --verbosity 5
```
Or access the jupyter notebooks:
```
jupyter lab
```
