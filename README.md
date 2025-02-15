Compute the lowest k eigenvalues of phase space model in Xuezhi's paper using the Davidson method with preconditioner.

```
# set up the environment
source env.modules

for g in 4; do
  for M in {2,5,10,20,40}; do
    qsub -j oe -N davidson-${g}-${M} -V -l ncpus=48 \
      -- $(which python) $(readlink -f davidson.py) \
      -g $g -M $M -r 400 -R 101 --iterations 10000 \
      --verbosity 5 --save Exact_c${g}m${M}.dat
  done
done
```

To setup on jupiter:

```
# set up the environment
source env.modules

# first install openssl

git clone "git@github.com:openssl/openssl.git"
cd openssl
./Configure --prefix=$HOME/.local
make -j 48
make install

# Now setup python
## First pyenv

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
eval "$(pyenv init - zsh)"

CONFIGURE_OPTS="--with-openssl=$HOME/.local" pyenv install 3.12.9
pyenv local 3.12.9

## install other packages

MAKEFLAGS="-j48" pip install numpy --no-binary numpy
MAKEFLAGS="-j48" pip install scipy --no-binary scipy
MAKEFLAGS="-j48" pip install pyscf --no-binary pyscf
```
