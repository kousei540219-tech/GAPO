# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import collections  


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "mountaincar"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MountainCarContinuous-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    num_groups: int = 1
    """number of environment groups used for the group-relative (same-seed) baseline"""
    aux_coef: float = 1.0
    """weight of the episodic group-relative auxiliary term"""
    solved_return: float = 90
    """episodic return threshold counted as a 'solved' episode (for success_rate)"""
    return_window: int = 100
    """number of recent episodes used for rolling avg_return/avg_length and success_rate"""
    target_minibatch_size: int = 256
    """fixed mini-batch size, kept constant regardless of the varying episode-mode batch size"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
def log_stats(infos, writer, global_step, return_deque, length_deque, solved_return):
    # gymnasium >=1.0 vector API: completed-episode stats live under
    # infos["episode"] as arrays, masked by infos["_episode"].
    if "episode" not in infos:
        return
    ep = infos["episode"]
    mask = infos.get("_episode", np.ones_like(ep["r"], dtype=bool))
    for r, l, done in zip(ep["r"], ep["l"], mask):
        if done:
            ep_return = float(r)
            ep_length = int(l)
            print(f"global_step={global_step}, episodic_return={ep_return:.1f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            return_deque.append(ep_return)
            length_deque.append(ep_length)
            writer.add_scalar("charts/avg_return", float(np.mean(return_deque)), global_step)
            writer.add_scalar("charts/avg_length", float(np.mean(length_deque)), global_step)
            success_rate = float(np.mean([x >= solved_return for x in return_deque]))
            writer.add_scalar("charts/success_rate", success_rate, global_step)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

        # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, Args.gamma) for i in range(args.num_envs)],
    )

    assert args.num_groups > 0, "num_groups must be > 0"
    assert args.num_envs % args.num_groups == 0, \
        f"num_envs ({args.num_envs}) must be divisible by num_groups ({args.num_groups})"
    env_groups = [list(range(args.num_envs))[i::args.num_groups] for i in range(args.num_groups)]
    print(f"Env Groups: {env_groups}")

    agent = Agent(envs).to(device)
    dtype = next(agent.parameters()).dtype
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=dtype, device=device)
    return_deque = collections.deque(maxlen=args.return_window)
    length_deque = collections.deque(maxlen=args.return_window)

    iteration = 0
    while global_step < args.total_timesteps:   # episode mode: 実ステップで判定
        iteration += 1
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - global_step / args.total_timesteps   # 実ステップ進捗で減衰
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ===== STAGE 1: DATA COLLECTION (one full episode per env) =====
        t_collection_start = time.perf_counter()

        # Reset every env; envs within a group share a seed so their episodes
        # start from identical conditions, making the group baseline comparable.
        group_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_groups)]
        for group_idx, env_idxs in enumerate(env_groups):
            for env_idx in env_idxs:
                _obs, _ = envs.envs[env_idx].reset(seed=group_seeds[group_idx])
                next_obs[env_idx] = torch.tensor(_obs, dtype=dtype, device=device)

        # We collect per env first so indexes match between obs, actions, and returns across episodes
        per_env_obs = [[] for _ in range(args.num_envs)]
        per_env_actions = [[] for _ in range(args.num_envs)]
        per_env_logprobs = [[] for _ in range(args.num_envs)]
        per_env_rewards = [[] for _ in range(args.num_envs)]
        per_env_values = [[] for _ in range(args.num_envs)]   # critic V(s_t) per step (Path A)
        per_env_final_obs = [None] * args.num_envs   # observation of s_final (for truncation bootstrap)
        per_env_truncated = [False] * args.num_envs  # True if the episode ended by time-limit truncation
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

        while not bool(finished.all()):
            # Count only steps from envs still collecting
            global_step += int((~finished).sum().item())
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                value = value.flatten()
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            reward_t = torch.tensor(reward, dtype=dtype, device=device).view(-1)
            final_obs = torch.tensor(next_obs_np, dtype=dtype, device=device)

            # Record for envs still collecting (obs/action that produced this reward)
            for i in range(args.num_envs):
                if not bool(finished[i]):
                    per_env_obs[i].append(next_obs[i].detach())
                    per_env_actions[i].append(action[i].detach())
                    per_env_logprobs[i].append(logprob[i].detach())
                    per_env_rewards[i].append(reward_t[i].detach())
                    per_env_values[i].append(value[i].detach())
                    if bool(done[i]):
                        finished[i] = True
                        # Save s_final and how the episode ended so the GAE stage can
                        # bootstrap V(s_final) on time-limit truncation. Under NEXT_STEP
                        # autoreset, final_obs[i] is the genuine final observation.
                        per_env_final_obs[i] = final_obs[i].detach()
                        per_env_truncated[i] = bool(truncations[i]) and not bool(terminations[i])

            log_stats(infos, writer, global_step, return_deque, length_deque, args.solved_return)
            next_obs = final_obs

        # Build flat batch tensors for episode mode
        b_obs_list = []
        b_actions_list = []
        b_logprobs_list = []
        per_env_rewards_list = []  # Keep raw per-env rewards for return calculation
        per_env_values_list = []
        for i in range(args.num_envs):
            b_obs_list.extend(per_env_obs[i])
            b_actions_list.extend(per_env_actions[i])
            b_logprobs_list.extend(per_env_logprobs[i])
            per_env_rewards_list.append([float(r) for r in per_env_rewards[i]])
            per_env_values_list.append([float(v) for v in per_env_values[i]])

        b_obs = torch.stack(b_obs_list, dim=0)
        b_actions = torch.stack(b_actions_list, dim=0)
        b_logprobs = torch.stack(b_logprobs_list, dim=0)
        t_collection_end = time.perf_counter()

        # ===== STAGE 2: RETURN AND AGE CALCULATION =====
        t_preprocessing_start = time.perf_counter()
        b_returns_list = []
        per_env_returns = []
        # Monte Carlo Returns
        for env_rewards in per_env_rewards_list:
            R = 0.0
            env_returns = []
            for r in reversed(env_rewards):
                R = float(r) + args.gamma * R
                env_returns.append(R)
            env_returns.reverse()
            per_env_returns.append(env_returns)
            b_returns_list.extend(env_returns)
        b_returns = torch.tensor(b_returns_list, dtype=dtype, device=device)
        #AGE
        b_advantage_gae_list = []
        b_returns_gae_list = []
        b_values_list = []
        for env_idx, (env_rewards, env_values) in enumerate(zip(per_env_rewards_list, per_env_values_list)):
            L = len(env_rewards)
            adv = [0.0] * L
            lastgaelam = 0.0
            for t in reversed(range(L)):
                if t == L - 1:
                    if per_env_truncated[env_idx]:
                        with torch.no_grad():
                            nextvalue = agent.get_value(per_env_final_obs[env_idx].unsqueeze(0)).item()
                    else:
                        nextvalue = 0.0
                else:
                    nextvalue = env_values[t +1]
                delta = env_rewards[t] + args.gamma * nextvalue - env_values[t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda  * lastgaelam
            b_advantage_gae_list.extend(adv)
            b_returns_gae_list.extend(adv[k] + env_values[k] for k in range(L))
            b_values_list.extend(env_values)
        b_advantages_gae = torch.tensor(b_advantage_gae_list, dtype=dtype, device=device)
        b_returns_gae    = torch.tensor(b_returns_gae_list, dtype=dtype, device=device)
        b_values         = torch.tensor(b_values_list,  dtype=dtype, device=device)
        
            




        # ===== STAGE 3: BASELINE CALCULATION (UNIFIED) =====
        # Calcucale Auxiliary term
        mean_vals = 0
        for episode_return ,*_ in per_env_returns:
            mean_vals += episode_return
        mean_vals = mean_vals / args.num_envs




        Auxiliary_vals = []
        for env_idx in range(args.num_envs):
            episode_return = per_env_returns[env_idx][0]
            aux = episode_return - mean_vals
            Auxiliary_vals.extend([aux] * len(per_env_returns[env_idx]))
        Auxiliary_vals = torch.tensor(Auxiliary_vals,dtype = dtype, device = device)



        # Calculate mean over steps for each env in a group, then mean over the group
        env_baseline = {}
        for group_idx, env_idxs in enumerate(env_groups):
            env_means = [np.mean(per_env_returns[env_idx]) for env_idx in env_idxs]
            group_baseline = float(np.mean(env_means))
            for env_idx in env_idxs:
                env_baseline[env_idx] = group_baseline
            
        baseline_vals = []
        for env_idx in range(args.num_envs):
            baseline_vals.extend([env_baseline[env_idx]] * len(per_env_returns[env_idx]))
            
        baseline_vals = torch.tensor(baseline_vals, dtype=dtype, device=device)
        b_advantages_gae /= (b_advantages_gae.std()+ 1e-8) 
        Auxiliary_vals /= (Auxiliary_vals.std() + 1e-8)
        b_advantages = b_advantages_gae + Auxiliary_vals * args.aux_coef

                # --- 2種類のadvantageを可視化 ---
        writer.add_scalar("adv/gae_mean", b_advantages_gae.mean().item(), global_step)
        writer.add_scalar("adv/gae_std",  b_advantages_gae.std().item(),  global_step)
        writer.add_scalar("adv/aux_mean", Auxiliary_vals.mean().item(),   global_step)
        writer.add_scalar("adv/aux_std",  Auxiliary_vals.std().item(),    global_step)
        writer.add_scalar("adv/aux_scaled_mean", (Auxiliary_vals * args.aux_coef).mean().item(), global_step)
        writer.add_scalar("adv/combined_mean", (b_advantages_gae + Auxiliary_vals * args.aux_coef).mean().item(), global_step)

        # 補助項がGAEに対してどれくらいの割合で効いているか
        writer.add_scalar("adv/aux_to_gae_ratio",
            (Auxiliary_vals.abs().mean() * args.aux_coef / (b_advantages_gae.abs().mean() + 1e-8)).item(),
            global_step)

        # --- 学習中のエピソード（各envの1エピソードのreturnとlength）---
        ep_returns = [per_env_returns[i][0] for i in range(args.num_envs)]   # 割引リターン
        ep_lengths = [len(per_env_returns[i]) for i in range(args.num_envs)]
        writer.add_scalar("episode/return_mean", float(np.mean(ep_returns)), global_step)
        writer.add_scalar("episode/return_std",  float(np.std(ep_returns)),  global_step)
        writer.add_scalar("episode/length_mean", float(np.mean(ep_lengths)), global_step)
        writer.add_histogram("episode/returns", np.array(ep_returns), global_step)  # 分布も見たいなら

        # Final batch setup
        batch_size = b_obs.shape[0]
        # Episode-mode batch size varies every iteration (= sum of episode lengths),
        # so fix the minibatch size instead of the number of minibatches. This keeps
        # the gradient-step granularity stable and scales #updates with the batch.
        minibatch_size = int(max(1, min(args.target_minibatch_size, batch_size)))

        t_preprocessing_end = time.perf_counter()



        # ===== STAGE 4: TRAINING =====
        t_update_start = time.perf_counter()
        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        grad_norms = []
        policy_ratios = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    policy_ratios.append(ratio.mean().item())

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns_gae[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns_gae[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns_gae[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                grad_norms.append(grad_norm.item())
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break


        t_update_end = time.perf_counter()

        # Calculate timing metrics
        time_collection = t_collection_end - t_collection_start
        time_preprocessing = t_preprocessing_end - t_preprocessing_start
        time_update = t_update_end - t_update_start

        # Per-iteration progress (so the run is visibly not frozen)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/grad_norm", np.mean(grad_norms), global_step)
        writer.add_scalar("losses/policy_ratio", np.mean(policy_ratios), global_step)
        avg_return = (sum(return_deque) / len(return_deque)) if return_deque else float("nan")
        print(
            f"iter={iteration} global_step={global_step}/{args.total_timesteps} "
            f"SPS={sps} avg_return={avg_return:.1f}"
        )


        # Early stop on total timesteps
        if global_step >= args.total_timesteps:
            break

    envs.close()
    writer.close()