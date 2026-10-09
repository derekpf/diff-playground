# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)

A comprehensive suite of GPU-accelerated environments for robot learning research and sim-to-real, built with [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx).

## Choosing SoftJAX surrogate flags

Softness and ST settings belong to the user. Reward and event-history calls
honor `reward_st_enable`; sensor and physics ST settings remain independent.
Calibrate other optional flags against the backward quantity consumed by each
call site, checking direction and monotonicity across user settings. With ST
enabled, the forward must match the nominal operation. Composite gradients can
still change with ST because downstream operations see different input values.

- **Keep softness tied to physical units with `standardize=False`.** SoftJAX's
  default standardization normalizes and squashes values before computing soft
  selection weights, so `softness` no longer directly tracks a distance, height,
  or force scale. Leave standardization enabled when scale-normalized selection
  is intentional, and tune softness in that normalized domain.

- **Choose the selection-gradient rule to match the max contract.** For
  SoftJAX's selection-based `max`, `gated_grad=True` differentiates through the
  soft index and gives the exact derivative of the relaxed weighted-value
  output. `False` stops that path, leaving soft-index weights as the gradient.
  Those weights form a convex combination of hard-max subgradients. Choose
  `False` when that monotone max-like backward is the intended surrogate; choose
  `True` when the desired gradient is the exact derivative of the relaxed
  weighted-value output. Keep the choice fixed while comparing ST modes: ST
  controls the forward value, not the selection-gradient rule.

- **Check the complete backward path.** Probe ties, thresholds, and realistic
  input gaps. Measure both the operator Jacobian and the reward or state
  Jacobian that consumes it. A nonzero local gradient alone is insufficient.
  Verify nominal forward parity separately when ST is enabled.

For example, Go1's swing-peak maximum uses height in meters, so
`standardize=False` keeps softness on the native scale. At a height 1 cm below
the peak, `standardize=False` alone gave gradient `0.401`; also disabling
`gated_grad` gave `0.450`, so that point alone does not justify the latter. The
max contract gives a separate reason: over 201 ST-mode candidate gaps from
`-0.10` to `+0.10` m at softness `0.05`, `gated_grad=True` produced gradient
components outside `[0, 1]` at 74 points (range `[-0.0908, 1.0908]`), while
`False` stayed in `[0.1192, 0.8808]`; both sums were one. Since increasing either
candidate must not reduce a maximum, Go1 uses `False` for its swing-peak
backward. The gradient choice has the same monotonicity in both ST modes. This
local check does not determine the best gradient width or establish that the
resulting touchdown reward gradient is useful.

Running peaks, progress floors, height caps, and instantaneous reward extrema
use convex max/min selection weights so each candidate has a nonnegative local
derivative. Their downstream reward derivatives still need separate checks:
the local max/min condition does not establish a useful policy gradient.
Contact event comparisons and arithmetic resets pass gradients through the
sensor gap, event gate, flight history, and touchdown reward. Their relevant
metric is the Jacobian of that complete chain over nearby timesteps, including
its sign and width. Go1 checks include nonzero touchdown reward derivatives,
flight-history derivatives reaching a following touchdown reward, and
below-peak foot-height derivatives pointing toward the reward's target.
History checks cover Go1, G1, Spot, T1, H1, Berkeley, and Barkour. These checks
isolate environment bookkeeping by replacing the physics step with identity;
they do not establish full rollout gradients through the dynamics solver.

Other optional flags were checked by their local backward contract. The
default `gated=False` for ReLU and clip yields derivatives in `[0, 1]` across a
boundary; gated ReLU at `x=-2*softness` has derivative `-0.091`. Product-based
`any` and `all` retain unit sensitivity to one decisive input at the all-off
or all-on boundary; geometric-mean reduction divides that sensitivity by the
number of inputs. These defaults therefore stay unchanged.

Features include:

- Classic control environments from `dm_control`.
- Quadruped and bipedal locomotion environments.
- Non-prehensile and dexterous manipulation environments.
- Vision-based support available via the [MJWarp Batch Renderer](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#batch-rendering).

For more details, check out the project [website](https://playground.mujoco.org/).

> [!NOTE]
> We now support training with both the MuJoCo MJX JAX implementation, as well as the [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) implementation at HEAD. See this [discussion post](https://github.com/google-deepmind/mujoco_playground/discussions/197) for more details.

## Installation

You can install MuJoCo Playground directly from PyPI:

```sh
pip install playground
```

> [!IMPORTANT]
> We recommend users to install [from source](#from-source) to get the latest features and bug fixes from MuJoCo.

### <a id="from-source">From Source</a>

> [!IMPORTANT]
> Requires Python 3.10 or later.

1. `git clone git@github.com:google-deepmind/mujoco_playground.git && cd mujoco_playground`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
3. Create a virtual environment: `uv venv --python 3.12`
4. Activate it: `source .venv/bin/activate`
5. Install CUDA 12 jax: `uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu. `unset LD_LIBRARY_PATH` may need to be run before running this command.
6. Install playground from source: `uv --no-config sync --all-extras`
7. Verify installation: `uv --no-config run python -c "import mujoco_playground; print('Success')"`
    * **Note**: Menagerie assets will be downloaded automatically the first time you load a locomotion or manipulation environment. You can trigger this with: `uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"`

## Getting started

### Running from CLI
For basic usage, navigate to the repo's directory, install [from source](#from-source) with `jax[cuda12]`, and run:

```bash
train-jax-ppo --env_name CartpoleBalance
```

To train with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp):

```bash
train-jax-ppo --env_name CartpoleBalance --impl warp
```

Or with `uv`:

```bash
uv --no-config run train-jax-ppo --env_name CartpoleBalance --impl warp
uv --no-config run train-rsl-ppo --env_name CartpoleBalance --impl warp
```

### Basic Tutorials
| Colab | Description |
|-------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb) | Introduction to the Playground with DM Control Suite |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb) | Locomotion Environments |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/manipulation.ipynb) | Manipulation Environments |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/vision.ipynb) | Vision Environments |

### Training Visualization

To interactively view trajectories throughout training with [rscope](https://github.com/Andrew-Luo1/rscope/tree/main), install it (`pip install rscope`) and run:

```
python learning/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
# In a separate terminal
python -m rscope
```

## FAQ

### How can I contribute?

Get started by installing the library and exploring its features! Found a bug? Report it in the issue tracker. Interested in contributing? If you are a developer with robotics experience, we would love your help—check out the [contribution guidelines](CONTRIBUTING.md) for more details.

### Reproducibility / GPU Precision Issues

Users with NVIDIA Ampere architecture GPUs (e.g., RTX 30 and 40 series) may experience reproducibility [issues](https://github.com/google-deepmind/mujoco_playground/issues/86) in mujoco_playground due to JAX’s default use of TF32 for matrix multiplications. This lower precision can adversely affect RL training stability. To ensure consistent behavior with systems using full float32 precision (as on Turing GPUs), please run `export JAX_DEFAULT_MATMUL_PRECISION=highest` in your terminal before starting your experiments (or add it to the end of `~/.bashrc`).

To reproduce results using the same exact learning script as used in the paper, run the brax training script which is available [here](https://github.com/google/brax/blob/1ed3be220c9fdc9ef17c5cf80b1fa6ddc4fb34fa/brax/training/learner.py#L1). There are slight differences in results when using the `learning/train_jax_ppo.py` script, see the issue [here](https://github.com/google-deepmind/mujoco_playground/issues/171) for more context.

## Citation

If you use Playground in your scientific works, please cite it as follows:

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```

## License and Disclaimer

The texture used in the rough terrain for the locomotion environments is from [Polyhaven](https://polyhaven.com/a/rock_face) and licensed under [CC0](https://creativecommons.org/public-domain/cc0/).

All other content in this repository is licensed under the Apache License, Version 2.0. A copy of this license is provided in the top-level [LICENSE](LICENSE) file in this repository. You can also obtain it from https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
