# GAPO: Enhancing Policy Optimization with Group Augmentation in Reinforcement Learning

GAPO (Group-Augmented PPO) adds two elements to Proximal Policy Optimization (PPO): a
**group rollout**, in which a group of `N` environments is reset to the same seed and each
runs one full episode per update, and a **group-relative advantage**, the episode's
discounted return minus the group mean, broadcast to every step of that episode. The
advantage can be **added** to the GAE advantage inside a single clipped surrogate (additive
variant) or trained through a **second** clipped surrogate with its own weight (separated
variant). Setting the weight to zero leaves the group rollout alone and gives the
`beta_group = 0` control used in the paper. All scripts are single-file modifications of
CleanRL's PPO implementations and share its network, optimizer and clipping settings.

## Publication

This repository accompanies the paper

> Kosei Eguchi and Donghui Lin.
> **GAPO: Enhancing Policy Optimization with Group Augmentation in Reinforcement Learning.**
> Okayama University. To appear.

The paper compares PPO, GRPO, the two GAPO variants and the `beta_group = 0` control on
CartPole-v1, Acrobot-v1, MountainCarContinuous-v0 and HalfCheetah-v5 with ten seeds each
(Table III), then sweeps the group size `N` (Table IV) and the weight `beta_group`
(Table V). If you use this code, please cite:

```bibtex
@inproceedings{Eguchi2026GAPO,
  author    = {Eguchi, Kosei and Lin, Donghui},
  title     = {{GAPO}: Enhancing Policy Optimization with Group Augmentation in Reinforcement Learning},
  booktitle = {the 2026 IEEE International Conference on Agentic AI (IEEE ICA 2026)},
  year      = {2026},
  note      = {To appear}
}
```

## Repository Layout

| Path | Contents |
| --- | --- |
| `ppo.py` | PPO baseline for discrete actions (CleanRL `ppo.py`; CartPole-v1, Acrobot-v1) |
| `ppo_continuous_action.py` | PPO baseline for continuous actions (CleanRL `ppo_continuous_action.py`; MountainCarContinuous-v0, HalfCheetah-v5) |
| `GAPO_additive.py` / `GAPO_continuous_additive.py` | GAPO (additive): group-relative advantage added to the GAE advantage, weight `--aux_coef` (= alpha in the paper) |
| `GAPO_separated.py` / `GAPO_continuous_separated.py` | GAPO (separated): second clipped surrogate on the group-relative advantage, weight `--aux_coef` (= beta_group); `--aux_coef 0.0` is the control |
| `GRPO.py` / `GRPO_continuous.py` | GRPO baseline: the group-relative advantage alone in the clipped surrogate (the critic is trained but unused) |
| `run_all.sh` | Runs the five Table III methods for one environment, ten seeds each |

Every script writes TensorBoard logs to `runs/<env>__<exp_name>__<seed>__<time>/`
(`charts/episodic_return` is the per-episode return used for all tables and figures) and
accepts `--track` to mirror them to Weights & Biases.

## Installation

Tested with Python 3.10 on CPU. Create an environment and install the pinned versions
used for the paper:

```
pip install -r requirements.txt
```

This installs `torch 2.12.0`, `gymnasium 1.2.3` with the classic-control and MuJoCo extras,
`mujoco 3.10.0`, `tyro 1.0.13`, `tensorboard 2.20.0`, `numpy 2.2.6` and `wandb 0.27.0`
(the last only for `--track`). HalfCheetah-v5 needs the MuJoCo extra; the three
classic-control tasks do not.

## Running the Experiments

Methods map to scripts and options as follows:

| Method | Script | Options |
| --- | --- | --- |
| PPO | `ppo.py` / `ppo_continuous_action.py` | CleanRL defaults (see the HalfCheetah note below) |
| GRPO | `GRPO.py` / `GRPO_continuous.py` | `--aux_coef` scales the loss (continuous version only; see Notes) |
| GAPO (additive) | `GAPO_additive.py` / `GAPO_continuous_additive.py` | `--aux_coef <alpha>` |
| GAPO (separated) | `GAPO_separated.py` / `GAPO_continuous_separated.py` | `--aux_coef <beta_group>`, `--num_envs <N>` |
| `beta_group = 0` control | `GAPO_separated.py` / `GAPO_continuous_separated.py` | `--aux_coef 0.0` |

Per-environment settings (Table I of the paper). `N` is the group size, which is
`--num_envs` in every group script; the weight is the value selected in Table V and is
used for both variants.

| Environment | `--env_id` | `--total_timesteps` | `N` | alpha = beta_group | Other options |
| --- | --- | --- | --- | --- | --- |
| CartPole | `CartPole-v1` | `1000000` | 4 | 0.8 | discrete scripts |
| Acrobot | `Acrobot-v1` | `500000` | 4 | 1.0 | discrete scripts |
| MtnCarCont | `MountainCarContinuous-v0` | `500000` | 4 | 0.01 | continuous scripts |
| HalfCheetah | `HalfCheetah-v5` | `30000000` | 128 | 0.01 | continuous scripts, `--learning_rate 2.5e-4 --num_envs 128` |

Shared hyperparameters (Table II): Adam with eps 1e-5, learning rate 2.5e-4 annealed
(3e-4 on MountainCarContinuous), clip 0.2, GAE lambda 0.95, value coefficient 0.5, value
clipping on, gradient norm 0.5, 4 epochs with entropy 0.01 on CartPole and Acrobot, 10 epochs
with entropy 0 on the continuous tasks, actor/critic MLP [64, 64] with tanh and orthogonal
initialization. These are the script defaults; only the options in the table above need to
be passed.

### Table III (method comparison)

`run_all.sh` runs PPO, GRPO, GAPO (additive), GAPO (separated) and the control for one
environment with seeds 1 to `n_seeds` (default 10), passing the options above:

```
./run_all.sh cartpole
./run_all.sh acrobot
./run_all.sh mountaincar
./run_all.sh halfcheetah
DRY_RUN=1 ./run_all.sh acrobot          # print the commands only
./run_all.sh acrobot 3 --total_timesteps 200000   # quick check: 3 seeds, shorter budget
```

The single commands it issues are, for example on Acrobot,

```
python ppo.py                                   --env_id Acrobot-v1 --total_timesteps 500000 --seed 1
python GRPO.py            --aux_coef 0.8        --env_id Acrobot-v1 --total_timesteps 500000 --seed 1
python GAPO_additive.py   --aux_coef 1.0        --env_id Acrobot-v1 --total_timesteps 500000 --seed 1
python GAPO_separated.py  --aux_coef 1.0        --env_id Acrobot-v1 --total_timesteps 500000 --seed 1
python GAPO_separated.py  --aux_coef 0.0 --exp_name GAPO_control --env_id Acrobot-v1 --total_timesteps 500000 --seed 1
```

### Table IV (group size `N`) and Table V (weight `beta_group`)

Both sweeps use the separated variant. The `N` sweep fixes `beta_group = 0.1` and varies
`--num_envs` over 4, 8, 16, 32, 64 and 128; the weight sweep fixes `N` at the Table I value and
varies `--aux_coef` over 0.01, 0.1, 0.4, 0.8 and 1.0. Budgets are those of Table I in both
sweeps.

```
# N sweep (CartPole shown; Acrobot: Acrobot-v1 / 500000; MountainCarContinuous: continuous script / 500000)
for N in 4 8 16 32 64 128; do
  python GAPO_separated.py --env_id CartPole-v1 --total_timesteps 1000000 --num_envs $N --aux_coef 0.1 --exp_name GAPO_ne$N --seed 1
done
# HalfCheetah N sweep
python GAPO_continuous_separated.py --env_id HalfCheetah-v5 --total_timesteps 30000000 --learning_rate 2.5e-4 --num_envs $N --aux_coef 0.1 --seed 1

# weight sweep (MountainCarContinuous shown; HalfCheetah adds --total_timesteps 30000000 --learning_rate 2.5e-4 --num_envs 128)
for B in 0.01 0.1 0.4 0.8 1.0; do
  python GAPO_continuous_separated.py --env_id MountainCarContinuous-v0 --total_timesteps 500000 --aux_coef $B --exp_name GAPO_ne$B --seed 1
done
```

### Evaluation protocol

Each run is a seed; every table reports the mean over ten seeds with the sample standard
deviation (ddof = 1). The per-seed value is the mean of `charts/episodic_return` over the
episodes in the final part of training:

- Table III: episodes whose `global_step` lies in the last 10% of the budget (step >= 900k on
  CartPole, >= 450k on Acrobot and MountainCarContinuous, >= 27M on HalfCheetah).
- Tables IV and V: the last 10% of each run's episode list.

A seed counts as solved when its Table III value exceeds the Gymnasium reward threshold (475,
-100 and 90). Significance in the text is Welch's t-test on the per-seed values; differences
in standard deviation use an exact permutation test over the 20 seeds. Learning curves are
means and +-1 s.d. over seeds of the same per-episode returns.

## About the Implementation

The group scripts replace CleanRL's fixed-length rollout with an episode-mode rollout:

1. Every `--num_envs` environments form one group (`--num_groups 1` in all paper runs). At the
   start of each update the group is reset with a shared seed and each environment runs one
   complete episode, so the batch is the concatenation of `N` whole episodes of varying length.
2. The GAE advantage is computed per step as in PPO, bootstrapping the value on time-limit
   truncation. The group-relative advantage of episode `i` is its discounted return minus the
   mean return of the group, broadcast to all its steps. Both advantages are divided by their
   batch standard deviation.
3. GAPO (separated) optimizes two clipped surrogates, one per advantage, and weights the
   second by `--aux_coef`. GAPO (additive) optimizes one clipped surrogate on the sum
   `GAE + aux_coef * group_relative`. GRPO uses the group-relative surrogate alone. The critic is
   trained on the GAE returns in all three.
4. Because the batch size varies between updates, the minibatch size is fixed at
   `min(--target_minibatch_size, batch)` with a default of 256, rather than fixing the number
   of minibatches as CleanRL does. All other settings (epochs, clipping, entropy, value loss,
   gradient clipping, learning-rate annealing) are CleanRL's.

The continuous scripts keep CleanRL's observation and reward normalization and action
clipping wrappers; the discrete scripts use none.

## Notes

- **HalfCheetah PPO.** The PPO run in Table III uses a rollout of 1000 steps per environment
  and 500 minibatches, so that its minibatch size is 256 like the group methods'
  (`run_all.sh halfcheetah` passes `--num_steps 1000 --num_minibatches 500`). The group
  methods use CleanRL's 256-step default, which in episode mode is only the buffer size.
- **GRPO weight.** `GRPO_continuous.py` multiplies the policy loss by `--aux_coef`
  (1.0 on MountainCarContinuous, 0.2 on HalfCheetah); `GRPO.py` does not. The value is a
  loss scale, not a mixing weight, since GRPO has no second advantage.
- **Determinism.** With `--torch_deterministic` (default) and a fixed seed, CPU runs are
  bit-reproducible; the shared group seed is drawn from Python's `random` after seeding.
- `GAPO_separated.py` and `GRPO.py` also contain a MiniGrid branch (image observations with a
  CNN encoder). It is not used by any paper experiment and is skipped for vector observations.

## License

The scripts derive from [CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT License); the
GAPO modifications are released under the same license.
