#!/usr/bin/env bash
# Reproduce the Table III runs of the paper (PPO, GRPO, GAPO additive, GAPO separated,
# and the beta_group = 0 control) for one environment, 10 seeds each.
#
# Usage:
#   ./run_all.sh <cartpole|acrobot|mountaincar|halfcheetah> [n_seeds] [extra args...]
#     n_seeds    : seeds 1..n_seeds are run for every method (default 10)
#     extra args : appended to every python call (e.g. --total_timesteps 200000)
#   DRY_RUN=1 ./run_all.sh acrobot        # print the commands without running them
#
# Notes:
#   - Requires the conda env with torch / gymnasium / tyro (and mujoco for HalfCheetah).
#   - Runs on CPU and logs to TensorBoard under runs/ (no wandb; add --track to enable it).
#   - The aux_coef values are the per-environment weights of Table I; the control is
#     GAPO_separated with --aux_coef 0.0. HalfCheetah PPO uses the rollout length and
#     minibatch count of the paper's run (num_steps 1000, 500 minibatches of 256).
#   - set -euo pipefail: the sweep stops at the first failing run.
set -euo pipefail
cd "$(dirname "$0")"

ENV="${1:-}"; N="${2:-10}"
if [[ $# -ge 2 ]]; then shift 2; elif [[ $# -ge 1 ]]; then shift 1; fi
EXTRA=("$@")

case "$ENV" in
  cartpole)
    COMMON=(--env_id CartPole-v1 --total_timesteps 1000000)
    RUNS=(
      "ppo.py"
      "GRPO.py --aux_coef 1.0"
      "GAPO_additive.py --aux_coef 0.8"
      "GAPO_separated.py --aux_coef 0.8"
      "GAPO_separated.py --aux_coef 0.0 --exp_name GAPO_control"
    ) ;;
  acrobot)
    COMMON=(--env_id Acrobot-v1 --total_timesteps 500000)
    RUNS=(
      "ppo.py"
      "GRPO.py --aux_coef 0.8"
      "GAPO_additive.py --aux_coef 1.0"
      "GAPO_separated.py --aux_coef 1.0"
      "GAPO_separated.py --aux_coef 0.0 --exp_name GAPO_control"
    ) ;;
  mountaincar)
    COMMON=(--env_id MountainCarContinuous-v0 --total_timesteps 500000)
    RUNS=(
      "ppo_continuous_action.py"
      "GRPO_continuous.py --aux_coef 1.0"
      "GAPO_continuous_additive.py --aux_coef 0.01"
      "GAPO_continuous_separated.py --aux_coef 0.01"
      "GAPO_continuous_separated.py --aux_coef 0.0 --exp_name GAPO_control"
    ) ;;
  halfcheetah)
    COMMON=(--env_id HalfCheetah-v5 --total_timesteps 30000000 --learning_rate 2.5e-4 --num_envs 128)
    RUNS=(
      "ppo_continuous_action.py --num_steps 1000 --num_minibatches 500"
      "GRPO_continuous.py --aux_coef 0.2"
      "GAPO_continuous_additive.py --aux_coef 0.01"
      "GAPO_continuous_separated.py --aux_coef 0.01"
      "GAPO_continuous_separated.py --aux_coef 0.0 --exp_name GAPO_control"
    ) ;;
  *)
    echo "usage: $0 <cartpole|acrobot|mountaincar|halfcheetah> [n_seeds] [extra args...]" >&2; exit 1 ;;
esac

mkdir -p runs
echo ">> $ENV: ${#RUNS[@]} methods x $N seeds  extra: ${EXTRA[*]:-(none)}"
for RUN in "${RUNS[@]}"; do
  read -r -a PARTS <<< "$RUN"
  echo
  echo "==================== ${PARTS[*]}  (seed 1..$N) ===================="
  for s in $(seq 1 "$N"); do
    CMD=(python "${PARTS[@]}" "${COMMON[@]}" --seed "$s" ${EXTRA[@]+"${EXTRA[@]}"})
    echo "----- seed $s / $N: ${CMD[*]}"
    if [[ -z "${DRY_RUN:-}" ]]; then "${CMD[@]}"; fi
  done
done
echo
echo ">> done. TensorBoard logs are under runs/ (tensorboard --logdir runs)"
