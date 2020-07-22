# IPO on Colored-Key example

Code for Colored-Key example in IPO paper. 

# Installation

First install the following packages (the code for the specific versions of these packages we used are provided in the repository containing the IPO code):

MiniGrid: https://github.com/maximecb/gym-minigrid

RL Starter Files: https://github.com/lcswillems/rl-starter-files

Torch-AC: https://github.com/lcswillems/torch-ac

Then, do the following:

```
cd Colored-Keys
pip3 install -e .
```

This adds MiniGrid environments with colored keys.

Finally, to ensure reproducibility and prevent issues with backwards compatibility, please install pytorch version 1.2.0 (newer versions of pytorch are not all backwards compatible and can also be different in terms of random seeds):

```
pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html

```

# Running the code

Execute the following:

```
cd rl-starter-files
python3 run_comparisons.py
```

This will run either IPO or PPO on some number of seeds (you can set the number of seeds by changing num_evals). You can change whether IPO or PPO is run by changing "method" to 'ipo' or 'ppo'. For each seed, the code performs both training (using red and green environments) and evaluation (on grey environments). The trained models are stored in the storage folder in rl-starter-files. The file multi_domain_train.py in rl-starter-files/scripts does the heavy lifting in terms of training. 

If you'd like to visualize a trained IPO policy, run the following from the rl-starter-files folder. This will visualize the policy corresponding to seed 0 on environments with Grey keys. You can change the --env argument to run the policy on a different domain (e.g., MiniGrid-ColoredKeysRed-v0).

```
python3 visualize.py --env MiniGrid-ColoredKeysGrey-v0 --model MiniGrid-ColoredKeysIPO-0 --ipo_model
```

To visualize a trained PPO model, run the following:

```
python3 visualize.py --env MiniGrid-ColoredKeysGrey-v0 --model MiniGrid-ColoredKeysPPO-0 
```

If you'd like to evaluate a trained IPO policy independently of run_comparisons.py (after you've executed run_comparisons.py once), you can execute the following from the rl-starter-files folder. 

```
python3 evaluate.py --env MiniGrid-ColoredKeysGrey-v0 --model MiniGrid-ColoredKeysIPO-0 --ipo_model
``` 

Similarly, you can evaluate a PPO model by running:

```
python3 evaluate.py --env MiniGrid-ColoredKeysGrey-v0 --model MiniGrid-ColoredKeysPPO-0
``` 



