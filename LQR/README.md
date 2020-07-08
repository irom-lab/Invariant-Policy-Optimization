# IPO on LQR systems

Code for testing IPO on Linear Quadratic Regulator problems. To compare IPO with gradient descent and overparameterization on multiple seeds, simply execute:

```
python3 run_comparisons.py. 
```

The number of domains used for training, the number of seeds, and the dimensionality of distractor observations can be changed in run_comparisons.py. 

Each script can be run directly from the command line as well. For example, in order to run IPO from the command line, execute the following:

```
python3 run_ipo_lqr.py --seed 0 --num_domains 2 --verbose 1 --num_distractors 1000
```
