# IPO on DoorGym

Code for door-opening example in IPO paper. This code makes use of the DoorGym environments:
https://github.com/PSVL/DoorGym

## Installation

First, follow the installation instructions for DoorGym (provided below) and download the doorknob datasets (we will be making use of the `pull knobs` and the `floating hook` end effector). 

Next, generate the training environments from two domains with differing values of door-hinge friction: 

```
cd path/to/DoorGym/world_generator

python3 custom_world_generator.py --knob-type pull --robot-type floatinghook --output-name-extension train1 --frictionloss-low 0.0 --frictionloss-high 0.0

python3 custom_world_generator.py --knob-type pull --robot-type floatinghook --output-name-extension train2 --frictionloss-low 0.1 --frictionloss-high 0.1
```

Next, generate environments for the test domain. The instruction below is for generating environments with frictionloss equal to 1.3. 

```
python3 custom_world_generator.py --knob-type pull --robot-type floatinghook --output-name-extension test --frictionloss-low 1.3 --frictionloss-high 1.3
```

Then, execute run_comparisons.py (located in the root folder of this repository). This will train PPO and IPO across some number of seeds (the number of seeds can be set in run_comparisons.py). Since training is time-consuming, we have also provided five models pre-trained using PPO and IPO. To use these models instead of training your own, skip the next step and see below. 

```
python3 run_comparisons.py
```

The trained models are stored in the `trained_models` folder. We have also provided models pre-trained using PPO and IPO in the `pretrained_models` folder. To evaluate a model on environments from the test domain, run the following:

```
python3 enjoy.py --env-name doorenv-v0 --load-name [pretrained_models or trained_models]/[ppo or ipo]/doorenv-v0_[ppo or ipo]-test-frictionloss[seed number].best.pt --world-path [absolute-path-to-DoorGym]/world_generator/world/pull_blue_floatinghooktest/ --eval
```

To visualize a policy on test environments, simply remove the `--eval` argument from the line above. 


# DoorGym

These instructions are reproduced from DoorGym.  

## 0. Set up the environment
requirement:

- Ubuntu 16.04 or later (Confirmed only on 16.04)

-Python3.6 or later

-Mujoco

-Mujoco-py

-Gym 0.14.0 or later

-OpenAI Baseline 0.1.6 or later

-Unity3D

etc.

### Conda (Anaconda, Miniconda)
#### Step1. Install Mujoco
1. Get the license from [MuJoCo License page](https://www.roboti.us/license.html)

2. Download Mujoco2.00 from [MuJoCo Product page](https://www.roboti.us/index.html).

3. Extruct it and place it in under `home/.mujoco`, as `/mujoco200` (Not mujoco200_linux).

4. Put your key under both `.mujoco/` and `.mujoco/mujoco200/bin`.

Detailed installation can be checked in following page.
https://github.com/openai/mujoco-py

#### Step2. Set environment var. and install necessary pkgs
```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev libopenmpi-dev patchelf
```
Set following path in `~/.bashrc`.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/[usr-name]/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-[driver-ver]
export PYTHONPATH=$PYTHONPATH:/home/[use-name]/DoorGym/DoorGym-Unity/python_interface
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

#### Step3. Clone DoorGym repo and Create conda environment
```bash
git clone https://github.com/PSVL/DoorGym
cd ./DoorGym
git submodule init
git submodule update
conda env create -n doorgym -f environment/environment.yml
conda activate doorgym
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

#### Step4. Install doorenv (0.0.1)
```bash
cd envs
pip install -e .
```

#### Step5. Install OpenAI Baselines (0.1.6<)
```bash
cd ../..
git clone https://github.com/openai/baselines
cd baselines
pip install -e .
```

#### Step6. Import test
Make sure that you can import pytorch and mujoco-py w/o problem.
```bash
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> import mujoco_py
```

If there is an error while importing mujoco-py, try trouble shooting according to the [mujoco-py installation guide](https://github.com/openai/mujoco-py)

## 1. Download the randomized door knob dataset
You can download from the following URL (All tar.gz file).
#### [Pull knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/pullknobs.tar.gz) (0.75 GB)
#### [Lever knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/leverknobs.tar.gz) (0.77 GB)
#### [Round knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/roundknobs.tar.gz) (1.24 GB)

* Extract and place the downloaded door knob dataset under the `world_generator/door` folder (or make a symlink).
* Place your favorite robots under the `world_generator/robot` folder. (Blue robots are there as default)



