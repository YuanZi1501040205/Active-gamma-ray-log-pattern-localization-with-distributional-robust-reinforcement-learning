# Active signal localization by using DR-SAC-Discrete in PyTorch

This repo aiming to combine human attention inspired Active Object Localization Computer vision technique and state-of-the-art safe reinforcement learning algorithm DR SAC(Distributional Robust Soft-Actor Critic)
 to solve the gamma-ray pattern matching problem. This is also the first application of DR-SAC with discrete action distribution.

**UPDATE**
- 2021.2.23
    - Create the repository.

## Setup
If you are using Anaconda, first create the virtual environment.

```bash
conda create -n sacd python=3.7 -y
conda activate sacd
```

You can install Python liblaries using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.


## Examples
You can train SAC-Discrete agent like this example.

```
python train.py
```

```
tensorboard --logdir=./logs/Gamma-Ray/sacd-seed0-20210209-2133
```

If you want to use Prioritized Experience Replay(PER), N-step return or Dueling Networks, change `use_per`, `multi_step` respectively.

## Results

## Code References

[[1]](https://github.com/ku2482/sac-discrete.pytorch) Soft Actor-Critic for Discrete Action

[[2]](https://github.com/bandofstraycats/dr-sac) Distributional Robust Soft Actor-Critic

## References
[[1]](https://arxiv.org/abs/1910.07207) Christodoulou, Petros. "Soft Actor-Critic for Discrete Action Settings." arXiv preprint arXiv:1910.07207 (2019).

[[2]](https://arxiv.org/abs/1902.08708) Smirnova, E., Dohmatob, E., & Mary, J. "Distributionally Robust Reinforcement Learning." (2019). 

[[3]](https://arxiv.org/abs/1511.05952) Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

[[4]](https://arxiv.org/abs/1511.06581) Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." arXiv preprint arXiv:1511.06581 (2015).
