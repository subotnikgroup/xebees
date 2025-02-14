Compute the lowest k eigenvalues of phase space model in Xuezhi's paper using the Davidson method with preconditioner.

```
for g in 4; do
  for M in {2,5,10,20,40}; do
    qsub -j oe -N davidson-${g}-${M} -V -l ncpus=48 \
      -- $(which python) $(readlink -f davidson.py) \
      -g $g -M $M -r 400 -R 101 --iterations 10000 \
      --verbosity 5 --save Exact_c${g}m${M}.dat
  done
done
```
