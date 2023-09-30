from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import torch
import torch.nn.functional as F

from pursuit_evasion_nn_agents import PursuitGraphModel
from pursuit_evasion_simulation import (EnvironmentObservation,
                                        PursuitGraphEvaderEnv)
from scenarios import Scenario


@dataclass
class Step:
    obs: EnvironmentObservation
    action: Any
    reward: float
    info: dict

@dataclass
class Episode:
    steps: List[Step]
    scenario: Scenario

    def get_discounted_episode(self, discount_factor: float) -> 'Episode':
        steps_with_discounted_rewards: List[Step] = []
        discounted_reward = 0
        for step in reversed(self.steps):
            discounted_reward = step.reward + discount_factor * discounted_reward
            steps_with_discounted_rewards.append(Step(step.obs, step.action, discounted_reward, step.info))
        steps_with_discounted_rewards.reverse()

        return Episode(steps=steps_with_discounted_rewards, scenario=self.scenario)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_as_policy(model: PursuitGraphModel, greedy=False):
    def wrapper(graph, evader_positions, pursuer_positions):
        my_position = next(iter(evader_positions.values()))
        output = model.networkx_forward(graph, my_position)
        logits = output.logits
        neighbor_nodes = output.neighbor_nodes

        if greedy:
            pos = torch.argmax(logits)
        else:
            pos = torch.argmax(torch.nn.functional.gumbel_softmax(logits, hard=True))
            
        neighbor_node = neighbor_nodes[pos]
        return neighbor_node

    return wrapper

def graphsage_contrastive_loss(embeddings, edge_index):
    # normalize embeddings
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    # (e, d)
    # left = embeddings[edge_index[0]]
    # (e, d)
    # right = embeddings[edge_index[1]]
    # (n, d) @ (d, n) = (n, n)
    loss = F.cross_entropy(embeddings @ embeddings.T, torch.arange(embeddings.shape[0])) # + F.cross_entropy(left @ right.T)
    # print(loss)
    return loss

def itemize_dict(d: dict):
    return {k: v.item() for k, v in d.items()}

def _get_logprob_and_value_for_action(graph: nx.Graph, evader_position, action, model: PursuitGraphModel):
    output = model.networkx_forward(graph, evader_position)
    logits = output.logits
    value = output.value
    neighbor_nodes = output.neighbor_nodes

    assert action in neighbor_nodes, f"Action not found in `neighbor_nodes`. Action: {action}. `neighbor_nodes`: {neighbor_nodes}"

    position = neighbor_nodes.index(action)
    log_prob = F.log_softmax(logits, dim=-1)[position]

    return (log_prob, value)

def train_step(model: PursuitGraphModel, optim: torch.optim.Optimizer, discounted_episodes: List[Episode], ppo_model: Optional[PursuitGraphModel] = None, use_advantage=True):
    optim.zero_grad()

    episode_losses: List[dict] = []
    total_discounted_rewards: List[float] = []

    for discounted_episode in discounted_episodes:
        discounted_reward_vec = torch.tensor([step.reward for step in discounted_episode.steps], device=device)
        log_prob_list = []
        value_list = []
        ppo_log_prob_list = []
        ppo_value_list = []
        # contrastive_loss_list = []

        model.train()

        for step in discounted_episode.steps:
            graph = step.obs.graph
            evader_position = next(iter(step.obs.evader_positions.values()))

            log_prob, value = _get_logprob_and_value_for_action(graph, evader_position, step.action, model)
            log_prob_list.append(log_prob)
            value_list.append(value)

            if ppo_model is not None:
                with torch.no_grad():
                    ppo_log_prob, ppo_value = _get_logprob_and_value_for_action(graph, evader_position, step.action, ppo_model)
                    ppo_log_prob_list.append(ppo_log_prob)
                    ppo_value_list.append(ppo_value)

        if ppo_model is None:
            log_prob_vec = torch.stack(log_prob_list)
            value_vec = torch.stack(value_list)

            # Accumulate loss
            loss_dict: Dict[str, torch.Tensor] = {}
            loss_dict['value'] = F.smooth_l1_loss(value_vec, discounted_reward_vec)
            if use_advantage:
                loss_dict['policy'] = (-log_prob_vec * (discounted_reward_vec - value_vec.detach())).mean()
            else:
                # use value estimate
                loss_dict['policy'] = (-log_prob_vec * (discounted_reward_vec)).mean()
        else:
            log_prob_vec = torch.stack(log_prob_list)
            value_vec = torch.stack(value_list)
            ppo_log_prob_vec = torch.stack(ppo_log_prob_list)
            ppo_value_vec = torch.stack(ppo_value_list)

            log_prob_ratio = log_prob_vec - ppo_log_prob_vec
            advantage = discounted_reward_vec - value_vec.detach()
            clipped_ratio = torch.clamp(log_prob_ratio, -0.2, 0.2)

            # Accumulate loss
            loss_dict: Dict[str, torch.Tensor] = {}
            loss_dict['value'] = F.smooth_l1_loss(value_vec, discounted_reward_vec)
            if use_advantage:
                loss_dict['policy'] = -torch.min(log_prob_ratio * advantage, clipped_ratio * advantage).mean()
            else:
                # use value estimate
                loss_dict['policy'] = -torch.min(log_prob_ratio * discounted_reward_vec, clipped_ratio * discounted_reward_vec).mean()
            
        episode_loss = sum(loss_dict.values())
        episode_loss = episode_loss / len(discounted_episodes)
        episode_loss.backward() # type: ignore
        episode_losses.append(itemize_dict(loss_dict))
        total_discounted_rewards.append(sum(step.reward for step in discounted_episode.steps))
    
    optim.step()

    return {"loss": episode_losses, "total_discounted_reward": total_discounted_rewards}

@torch.no_grad()
def sample_episode(env: PursuitGraphEvaderEnv, policy: Callable[[nx.Graph, dict, dict], Any]):
    steps: List[Step] = []

    obs = env.reset()
    done = False
    while not done:
        action = policy(obs.graph, obs.evader_positions, obs.pursuer_positions)
        next_obs, reward, done, info = env.step(action)

        steps.append(Step(obs, action, reward, info))
        obs = next_obs

    return steps

def sample_planned_episode(env: PursuitGraphEvaderEnv, path):
    steps: List[Step] = []

    obs = env.reset()
    done = False
    i = 0
    while not done:
        action = path[i]
        i += 1
        next_obs, reward, done, info = env.step(action)

        steps.append(Step(obs, action, reward, info))
        obs = next_obs

    return steps
