"""
Usage: train_for_deceptiveness [options]

Options:
    --graph_depth=<n>               Depth of the graph convolutional network [default: 2]
    --feed_forward_layers=<n>       Number of feed forward layers to use [default: 2]
    --deception_type=<type>         Type of deception function to use
    --model_type=<type>             Type of model to use; i.e. "sage", "tree_search", "mlp", "resnet"
    --distance_metric=<metric>      Type of distance to use [default: shortest_path:0]
    --num_batches=<n>               Number of batches to train for [default: 384]
    --copy_every=<n>                Copy PPO parameters every n iterations.
                                    If -1, disables PPO training. [default: 4]
    --benchmark_every=<n>           Benchmark every n iterations [default: 8]
    --render_every=<n>              Render every n iterations [default: 16]
    --dropout=<p>                   Dropout probability [default: 0.1]
    --num_prioritized_training_scenarios=<n>  Number of training scenarios for UCB.
                                              If -1, disables prioritized training. [default: 32]
    --num_prioritized_training_samples=<n>    Number of training samples for UCB
                                              (or per iteration when not using UCB). [default: 256]
    --num_benchmark_trials=<n>      Number of trials to run for benchmarking [default: 32]
    --custom_landscape=<landscape>  Use a custom landscape instead of the built-in ones [default: none]
    --continue_from_step=<step>     Use the corresponding model and optimizer from the given step.
                                    0 means no checkpoint is used. The most recent model
                                    from a folder with the same key is used. [default: 0]
    --seed=<n>                      Determines which random seed to use. Helpful when we want to run
                                    several trials deterministically. When using this option, a new
                                    folder will be created for every seed, so results can be aggregated
                                    and validated and a confidence interval of the learning curve can
                                    be created. [default: none]
    --custom_key=<key>              Allows you to specify a custom key for the experiment. Otherwise, a key
                                    will be formatted that contains all of the options. [default: none]
    --nullify_if_unsuccessful=<b>   Nullify deceptive rewards if we don't reach the goal [default: 0]
    --use_constant_deceptiveness=<b> Use a constant deceptiveness level instead of the goal rate [default: 0]
"""

import datetime
import itertools
import os
import random
import shutil
from collections import defaultdict
from typing import List, Optional

import cv2
import networkx as nx
import numpy as np
import torch
import tqdm
from docopt import docopt
from torch.utils.tensorboard.writer import SummaryWriter

import benchmarking
import pursuit_evasion_nn_agents as agents
from graph_rendering import render_pursuit_evasion_graph
from graph_rl import (Episode, Scenario, Step, model_as_policy, sample_episode,
                      train_step)
from landscape import Landscape, load_landscape, load_landscapes
from plotting import disable_plot_windows
from pursuit_evasion_simulation import create_environment_from_scenario
from scenarios import get_built_in_scenarios, get_random_scenario

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nullify_deceptive_bonus_if_unsuccessful(episode: Episode):
    # Nullifies deceptive rewards if we didn't reach the goal.
    reached_goal = episode.steps[-1].info['goal_bonus'] > 0
    if reached_goal:
        return episode
    else:
        # Nullify deception rewards
        new_steps = []
        for step in episode.steps:
            new_steps.append(Step(step.obs, step.action, 0, step.info))
        # Keep the penalty for not reaching the goal
        new_steps[-1].reward = episode.steps[-1].info['goal_bonus']

        return Episode(new_steps, episode.scenario)

def watch_episode(graph: nx.Graph, pos: dict, episode: Episode):
    for step in episode.steps:
        (evader_positions, pursuer_positions) = step.obs.evader_positions, step.obs.pursuer_positions
        evader_position = next(iter(evader_positions.values()))
        pursuer_positions = list(pursuer_positions.values())
        image = render_pursuit_evasion_graph(graph, pos, evader_position, pursuer_positions, node_size=10, edge_width=3, done=True)
        cv2.imshow("Watching Episode", image)
        cv2.waitKey(0)
    cv2.destroyWindow("Watching Episode")

def split_list(L: List, fraction, shuffle=True):
    if shuffle:
        random.seed(0)
        L = L.copy()
        random.shuffle(L)
    return L[:int(len(L) * fraction)], L[int(len(L) * fraction):]

class DeceptiveModelTrainer:
    def __init__(
        self,
        model: agents.PursuitGraphModel,
        learning_rate: float,
        weight_decay: float,
        train_landscapes: List[Landscape],
        movement_reward: float,
        evaluation_deception_levels: List[float],
        differentiate_deception: bool,
        deception_type: str,
        experiment_key: str,
        show_progress_bar: bool,
        deception_gamma: float,
        distance_metric: str,
        visibility: int,
        train_on_built_in_scenarios: bool,
        nullify_if_unsuccessful: bool,
        use_constant_deceptiveness: bool,
    ):
        self.movement_cost = movement_reward
        self.deception_type = deception_type
        self.differentiate_deception = differentiate_deception
        self.evaluation_deception_levels = evaluation_deception_levels
        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_key = experiment_key
        self.out_dir = f"models/{self.experiment_key}/{self.run_id}"
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.writer = SummaryWriter(self.out_dir + "/tensorboard")
        self.loss_history = []
        self.reward_history = []
        self.goal_reached_history = []
        self.goal_rate_k = 20
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.show_progress_bar = show_progress_bar
        self.train_landscapes = train_landscapes
        self.deception_gamma = deception_gamma
        self.distance_metric = distance_metric
        self.visibility = visibility
        self.step = 0
        self.train_on_built_in_scenarios = train_on_built_in_scenarios
        self.nullify_if_unsuccessful = nullify_if_unsuccessful
        self.use_constant_deceptiveness = use_constant_deceptiveness

    def progress_bar(self, iterable, **kwargs):
        if not self.show_progress_bar:
            return iterable
        else:
            return tqdm.tqdm(iterable, **kwargs)

    @property
    def goal_rate(self):
        if len(self.goal_reached_history) > 0:
            return float(np.mean(self.goal_reached_history[-self.goal_rate_k:]))
        else:
            return 0
        
    def get_user_as_policy(self, scenario: Scenario):
        # to_256 = lambda color: tuple([int(x * 255) for x in color[:3]])
        # colormap = cm.get_cmap("viridis")
        path = [scenario.start_position]
        # rewards = []

        def policy(graph: nx.Graph, evader_positions, pursuer_positions):
            evader_position = next(iter(evader_positions.values()))
            pursuer_positions = list(pursuer_positions.values())
            image = render_pursuit_evasion_graph(graph, scenario.landscape.layout, evader_position, pursuer_positions, node_size=10, edge_width=3, done=True)
            cv2.imshow("Scene", image)
            key = cv2.waitKey(0)
            y, x = path[-1]
            if key == ord('w'):
                path.append((y - 1, x))
            elif key == ord('a'):
                path.append((y, x - 1))
            elif key == ord('d'):
                path.append((y, x + 1))
            elif key == ord('s'):
                path.append((y + 1, x))
            elif key == ord('r'):
                return
            elif key == ord('q'):
                exit()
            if path[-1] not in graph.nodes:
                exit()
            # rewards.append(env.path_bonus_function(path))
            # print("Reward:", rewards[-1], "Total:", sum(rewards), "Path length:", len(path))
            return path[-1]
        
        return policy

    def log_loss(self, loss_dict):
        total = sum(loss_dict.values())
        self.writer.add_scalar('Loss/total', total, self.step)
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.step)

    def log_reward_metrics(self, batch: List[Episode]):
        # Take absolute value of deceptiveness so we scale according to the magnitude at which the deception function was applied.
        path_bonus_sum = np.array([sum(s.info['path_bonus'] for s in episode.steps)/(1e-5 + abs(episode.scenario.deceptiveness)) for episode in batch]).mean()
        last_step_reward = np.array([episode.steps[-1].reward for episode in batch]).mean()
        goal_reached = np.array([episode.steps[-1].info['goal_bonus'] > 0 for episode in batch]).mean()
        total_reward = np.array([sum(s.reward for s in episode.steps) for episode in batch]).mean()

        self.writer.add_scalar('Reward/path_bonus_sum', path_bonus_sum, self.step)
        self.writer.add_scalar('Reward/goal_reached', goal_reached, self.step)
        self.writer.add_scalar('Reward/last_step_reward', last_step_reward, self.step)
        self.writer.add_scalar('Reward/total_reward', total_reward, self.step)

    def update_parameters(self, batch: List[Episode], discount_gamma: float = 0.99, ppo_model: Optional[agents.PursuitGraphModel] = None):
        if self.nullify_if_unsuccessful:
            batch = [nullify_deceptive_bonus_if_unsuccessful(episode) for episode in batch]
        batch = [episode.get_discounted_episode(discount_gamma) for episode in batch]
        return train_step(self.model, self.optimizer, batch, ppo_model=ppo_model)

    def sample_batch(self, batch_size: int):
        episodes: List[Episode] = []
        for _ in range(batch_size):
            scenario = self.get_random_appropriate_training_scenario()
            episode = self.sample_training_episode(scenario)
            episodes.append(episode)
        return episodes

    def sample_prioritized_batch(self, num_scenarios, num_samples):
        # Try overfitting to base scenario
        if self.train_on_built_in_scenarios:
            scenarios = list(itertools.chain(*[
                get_built_in_scenarios(landscape, deceptiveness=0, movement_cost=self.movement_cost, deceptiveness_fraction=0.0)
                for landscape in self.train_landscapes
            ]))
        else:
            scenarios = [self.get_random_appropriate_training_scenario() for _ in range(num_scenarios)]
        visit_counts = np.zeros(num_scenarios)
        value_sums = np.zeros(num_scenarios)
        value_scores = None
        for i in range(num_samples):
            # Calculate upper confidence bound from AlphaZero
            # https://joshvarty.github.io/AlphaZero/
            total_visits = i + 1
            prior_scores = np.sqrt(total_visits) / (visit_counts + 1)
            value_scores = value_sums / (visit_counts + 1e-5)
            # In our case, we want to pick the case with the lowest value and lowest visit count
            ucb_scores = prior_scores + -value_scores
            index = np.argmax(ucb_scores)
            scenario = scenarios[index]
            episode = self.sample_training_episode(scenario)
            value_sums[index] += sum(step.reward for step in episode.steps)
            visit_counts[index] += 1
            yield episode

    def get_training_landscape(self):
        return random.choice(self.train_landscapes)

    def sample_training_episode(self, scenario: Scenario) -> Episode:
        env = create_environment_from_scenario(
            scenario,
            self.deception_type,
            self.differentiate_deception,
            self.deception_gamma,
            visibility=self.visibility,
            distance_metric=self.distance_metric
        )
        steps = sample_episode(env, model_as_policy(self.model))
        episode = Episode(
            steps=steps,
            scenario=scenario,
        )

        return episode
    
    def average_dicts(self, dicts: List[dict]):
        if len(dicts) == 0:
            return {}
        
        result = {}
        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v
        for k in result.keys():
            result[k] /= len(dicts)
        return result

    def train_prioritized(self, num_scenarios: int, num_samples: int, ppo_model: Optional[agents.PursuitGraphModel] = None):
        self.model.train()

        i = 0
        for episode in self.progress_bar(self.sample_prioritized_batch(num_scenarios, num_samples), total=num_samples):
            episodes = [episode]
            self.log_reward_metrics(episodes)
            train_result = self.update_parameters(episodes, discount_gamma=0.99, ppo_model=ppo_model)
            self.log_loss(self.average_dicts(train_result['loss']))
            self.writer.add_scalar("Reward/total_discounted_reward", train_result['total_discounted_reward'][0], self.step)
            self.goal_reached_history.extend([e.steps[-1].info['goal_bonus'] > 0 for e in episodes])
            self.step += len(episodes)

            i += 1

    def train(self, num_batches: int, batch_size: int, ppo_model: Optional[agents.PursuitGraphModel] = None):
        self.model.train()

        for batch_num in self.progress_bar(range(num_batches), desc='Training...'):
            episodes = self.sample_batch(batch_size)
            self.log_reward_metrics(episodes)
            train_result = self.update_parameters(episodes, discount_gamma=0.99, ppo_model=ppo_model)
            self.log_loss(self.average_dicts(train_result['loss']))
            self.writer.add_scalar("Reward/total_discounted_reward", train_result['total_discounted_reward'][0], self.step)
            self.goal_reached_history.extend([e.steps[-1].info['goal_bonus'] > 0 for e in episodes])
            self.step += len(episodes)

    def get_random_appropriate_training_scenario(self):
        landscape = random.choice(self.train_landscapes)
        if self.use_constant_deceptiveness:
            deceptiveness = 1
        else:
            deceptiveness = self.goal_rate
        movement_cost = self.movement_cost
        scenario = get_random_scenario(landscape, deceptiveness=deceptiveness, movement_cost=movement_cost)
        return scenario
    
    def benchmark(self, scenario: Scenario, num_trials: int, render: bool):
        self.model.eval()

        assert scenario.deceptiveness is None, "Scenario must have a blank deceptiveness level."
        
        episodes_by_key = benchmarking.get_benchmark_results(
            scenario,
            model_as_policy(self.model),
            self.evaluation_deception_levels,
            self.deception_type,
            self.differentiate_deception,
            self.deception_gamma,
            num_trials,
            visibility=self.visibility,
            distance_metric=self.distance_metric,
        )

        all_episodes: List[Episode] = []
        for key, episodes in episodes_by_key.items():
            all_episodes.extend(episodes)

        # Logging results
        goal_rate = np.array([e.steps[-1].info['goal_bonus'] > 0 for e in all_episodes]).mean()
        path_bonus = np.array([sum(s.info['path_bonus'] for s in e.steps)/(1e-5 + e.scenario.deceptiveness) for e in all_episodes]).mean()
        print("Held out scenario:", goal_rate, np.mean(path_bonus))

        self.writer.add_scalar(f"Held-out goal rate/{scenario.name}", goal_rate, self.step)
        self.writer.add_scalar(f"Held-out path bonus/{scenario.name}", path_bonus, self.step)

        # Rendering
        if render:
            benchmarking.render_benchmark_results(scenario, episodes_by_key, "Deceptiveness", self.out_dir, self.step)

        return (goal_rate, path_bonus)

def stratified_split(landscapes: List[Landscape], fraction: float):
    landscapes = landscapes.copy()
    random.seed(0)
    random.shuffle(landscapes)
    sorted_landscapes = defaultdict(list)
    # Get totals first
    for landscape in landscapes:
        landscape_size = landscape.name[:-1].split("x")
        landscape_size = (int(landscape_size[0]), int(landscape_size[1]))
        sorted_landscapes[landscape_size].append(landscape)
    # Then split
    split_landscapes = {
        size: split_list(landscapes, fraction, shuffle=False)
        for size, landscapes in sorted_landscapes.items()
    }
    train_landscapes = []
    test_landscapes = []
    for size, (train, test) in split_landscapes.items():
        train_landscapes.extend(train)
        test_landscapes.extend(test)
    return train_landscapes, test_landscapes

def main():
    disable_plot_windows()

    arguments = docopt(__doc__)
    model_type = arguments['--model_type']
    graph_depth = int(arguments['--graph_depth'])
    deception_type = arguments['--deception_type']
    feed_forward_layers = int(arguments['--feed_forward_layers'])
    distance_metric = arguments['--distance_metric']
    num_batches = int(arguments['--num_batches'])
    copy_every = int(arguments['--copy_every'])
    benchmark_every = int(arguments['--benchmark_every'])
    render_every = int(arguments['--render_every'])
    dropout = float(arguments['--dropout'])
    num_prioritized_training_scenarios = int(arguments['--num_prioritized_training_scenarios'])
    num_prioritized_training_samples = int(arguments['--num_prioritized_training_samples'])
    num_benchmark_trials = int(arguments['--num_benchmark_trials'])
    custom_key = arguments['--custom_key']
    if custom_key == 'none':
        custom_key = None
    seed = arguments['--seed']
    if seed == 'none':
        seed = None
    else:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    nullify_if_unsuccessful = bool(int(arguments['--nullify_if_unsuccessful']))
    use_constant_deceptiveness = bool(int(arguments['--use_constant_deceptiveness']))

    print(arguments)

    keys = [
        'evader_visited',
        'goal_distance',
        'decoy_distance',
        'remaining_time',
    ]

    MOVEMENT_COST = 0

    if model_type == 'sage':
        # For backwards compatibility
        model_type = 'gnn.sage'
    
    if model_type.startswith("gnn."):
        modeling_approach = model_type[4:]
        create_model = lambda: agents.PursuitGraphModel(
            dim=64,
            layers=feed_forward_layers,
            keys=keys,
            augment=False,
            dropout=dropout,
            modeling_approach=modeling_approach,
        )
    elif model_type == 'tree_search':
        create_model = lambda: agents.PursuitTreeSearchModel(
            dim=64,
            feed_forward_layers=feed_forward_layers,
            graph_depth=graph_depth,
            dropout=dropout,
            keys=keys,
            augment=False,
        ).to(device)
    elif model_type == 'mlp':
        create_model = lambda: agents.PursuitGridMLPModelSeparateNetworks(
            dim=64 * 4,
            feed_forward_layers=feed_forward_layers,
            graph_depth=graph_depth,
            dropout=dropout,
            keys=keys,
            augment=False,
        ).to(device)
    elif model_type == 'resnet':
        create_model = lambda: agents.PursuitResNetModel(
            channels=64,
            blocks=feed_forward_layers,
            graph_depth=graph_depth,
            keys=keys,
            augment=False,
        )
    else:
        raise ValueError("model_type is not supported: " + model_type)

    start_step = int(arguments['--continue_from_step'])
    if custom_key is None:
        experiment_key = f'v27_trials/{model_type}/{deception_type}/{distance_metric=}_{graph_depth=}_{feed_forward_layers=}'
    else:
        experiment_key = custom_key
    # experiment_key = f'v26.1_geometric_distance_overfit_test/{model_type}/{deception_type}/{distance_metric=}_{graph_depth=}_{feed_forward_layers=}'
    if seed is not None:
        experiment_key += f'_seed={seed}'
    start_iteration = start_step // 256

    model = create_model()
    if start_step > 0:
        # Find latest folder in `experiment_key`
        latest_folder = max(os.listdir(f"models/{experiment_key}"))
        checkpoint = f"models/{experiment_key}/{latest_folder}"
        print("Starting from checkpoint", checkpoint, "at iteration", start_step)
        model.load_state_dict(torch.load(checkpoint + "/model.pt"))
        # copy model to model_{start_iteration}.pt and optimizer to optimizer_{start_iteration}.pt
        shutil.copyfile(f"{checkpoint}/model.pt", f"{checkpoint}/model_{start_step}.pt")
        shutil.copyfile(f"{checkpoint}/optimizer.pt", f"{checkpoint}/optimizer_{start_step}.pt")
    else:
        checkpoint = None

    if copy_every > 0:
        ppo_model = create_model()
        ppo_model.load_state_dict(model.state_dict())
    else:
        ppo_model = None

    custom_landscape = arguments['--custom_landscape'] # 'square_rect_based/16x16A'
    if custom_landscape and custom_landscape != 'none':
        train_landscapes = [load_landscape(f'gridworlds/{custom_landscape}.tmx', vf_movement_penalty=MOVEMENT_COST, vf_goal_initial_value=1)]
        test_landscapes = train_landscapes
    else:
        folder = 'gridworlds/square_rect_based'
        landscapes = load_landscapes(folder=folder, vf_movement_penalty=MOVEMENT_COST, vf_goal_initial_value=1)
        if 'square_rect_based' in folder:
            train_landscapes, test_landscapes = stratified_split(landscapes, 0.6)
        else:
            train_landscapes, test_landscapes = split_list(landscapes, 0.7, shuffle=True)
    
    trainer = DeceptiveModelTrainer(
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        train_landscapes=train_landscapes,
        movement_reward=MOVEMENT_COST,
        evaluation_deception_levels=[1],
        differentiate_deception=False,
        deception_type=deception_type,
        experiment_key=experiment_key,
        show_progress_bar=True,
        deception_gamma=0.95,
        distance_metric=distance_metric,
        visibility=graph_depth,
        train_on_built_in_scenarios=False,
        nullify_if_unsuccessful=nullify_if_unsuccessful,
        use_constant_deceptiveness=use_constant_deceptiveness,
    )
    trainer.step = start_step

    if checkpoint is not None:
        trainer.optimizer.load_state_dict(torch.load(checkpoint + "/optimizer.pt"))

    test_scenarios = []
    for landscape in test_landscapes:
        test_scenarios.extend(
            get_built_in_scenarios(landscape, deceptiveness=None, movement_cost=MOVEMENT_COST, deceptiveness_fraction=0.5)
        )

    try:
        for iteration in range(start_iteration, num_batches):
            if num_prioritized_training_scenarios > 0:
                trainer.train_prioritized(
                    num_scenarios=num_prioritized_training_scenarios,
                    num_samples=num_prioritized_training_samples,
                    ppo_model=ppo_model # type: ignore
                )
            else:
                trainer.train(
                    num_batches=256,
                    batch_size=1,
                    ppo_model=ppo_model # type: ignore
                )

            # Copy PPO parameters every 4 iterations
            if copy_every > 0:
                if (iteration + 1) % copy_every == 0:
                    assert ppo_model, "PPO model must exist if `copy_every` is not -1."

                    ppo_model.load_state_dict(model.state_dict())
            else:
                assert ppo_model is None, "PPO model must not exist if `copy_every` is -1."
                
            # Benchmark every 4, render every 8.
            if (iteration + 1) % benchmark_every == 0:
                for scenario in test_scenarios:
                    trainer.benchmark(
                        scenario,
                        num_trials=num_benchmark_trials,
                        render=(iteration + 1) % render_every == 0
                    )
    except KeyboardInterrupt:
        pass

    base_dir = trainer.out_dir
    print("Saving model and optimizer.")
    torch.save(model.state_dict(), f"{base_dir}/model.pt")
    torch.save(trainer.optimizer.state_dict(), f"{base_dir}/optimizer.pt")
    print("Saved model and optimizer.")

if __name__ == '__main__':
    main()
