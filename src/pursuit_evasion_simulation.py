from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import gymnasium as gym
import networkx as nx

from deception_types import (AmbiguousDeceptionFunction, Differentiate,
                             ExaggerationDeceptionFunction,
                             MinimizeLikelihoodDeceptionFunction,
                             PathBonusFunction,
                             RewardMovingToGoalNondeceptiveFunction)
from graph_feature_engineering import (
    add_manhattan_distance_labels_to_networkx_graph,
    add_position_labels_to_networkx_graph,
    add_shortest_path_distance_labels_to_networkx_graph, add_uniform_attribute,
    manhattan_distance)

if TYPE_CHECKING:
    from scenarios import Scenario


class PursuitGraphSimulator:
    def __init__(self, graph: nx.Graph, evader_positions: dict, pursuer_positions: dict):
        self.graph = graph
        self._initial_positions = ({**evader_positions}, {**pursuer_positions})
        self.evader_positions = evader_positions
        self.pursuer_positions = pursuer_positions

    def reset(self):
        initial_evader_positions, initial_pursuer_positions = self._initial_positions
        self.evader_positions = {**initial_evader_positions}
        self.pursuer_positions = {**initial_pursuer_positions}

    def get_observation(self):
        return ({**self.evader_positions}, {**self.pursuer_positions})

    def step_pursuers(self, actions: dict):
        caught_evaders = set()

        for pursuer_id, next_pos in actions.items():
            assert pursuer_id in self.pursuer_positions, f"Pursuer `{pursuer_id}` does not exist."

            assert (
                self.graph.has_edge(self.pursuer_positions[pursuer_id], next_pos) or
                self.pursuer_positions[pursuer_id] == next_pos
            ), f"Target position {next_pos} is invalid for pursuer `{pursuer_id}` with position {self.pursuer_positions[pursuer_id]}"

            self.pursuer_positions[pursuer_id] = next_pos

            caught_evaders.update(
                [evader_id for evader_id, evader_pos in self.evader_positions.items() if evader_pos == next_pos]
            )
        
        for evader_id in caught_evaders:
            del self.evader_positions[evader_id]
        
        return caught_evaders

    def step_evaders(self, actions):
        caught_evaders = set()

        for evader_id, next_pos in actions.items():
            assert evader_id in self.evader_positions, f"Evader `{evader_id}` does not exist, or has been caught."

            assert (
                self.graph.has_edge(self.evader_positions[evader_id], next_pos) or
                self.evader_positions[evader_id] == next_pos
            ), f"Target position {next_pos} is invalid for evader `{evader_id}` with position {self.evader_positions[evader_id]}"

            self.evader_positions[evader_id] = next_pos

            if next_pos in self.pursuer_positions.values():
                caught_evaders.add(evader_id)
        
        for evader_id in caught_evaders:
            del self.evader_positions[evader_id]

        return caught_evaders

## Wrap the simulator in a Gym env. ##

## NOTE: This needs to be significantly updated before it can be used.

class PursuitGraphPursuerEnv(gym.Env):
    def __init__(self, simulator: PursuitGraphSimulator, pursuer_id, pursuer_policy_dict, evader_policy_dict):
        self.simulator = simulator
        self.pursuer_id = pursuer_id
        self.pursuer_policy_dict = pursuer_policy_dict
        self.evader_policy_dict = evader_policy_dict

        self.graph = simulator.graph
        self.num_evaders = len(simulator.evader_positions)
        self.num_pursuers = len(simulator.pursuer_positions)

    def reset(self):
        self.simulator.reset()
        return self.simulator.get_observation()

    def step(self, action):
        pursuer_actions = self._select_pursuer_actions()
        pursuer_actions[self.pursuer_id] = action
        caught_evaders = self.simulator.step_pursuers(pursuer_actions)

        # Check if game is over between these two steps. However, if the game is over, then
        # the evader actions will be no-ops.

        evader_actions = self._select_evader_actions()
        caught_evaders.update(self.simulator.step_evaders(evader_actions))
        
        terminated = len(self.simulator.evader_positions) == 0
        reward = len(caught_evaders)

        return self.simulator.get_observation(), reward, terminated, {}

    def _select_pursuer_actions(self):
        pursuer_actions = {}
        for pursuer_id, pursuer_policy in self.pursuer_policy_dict.items():
            if pursuer_policy is not None:
                pursuer_action = pursuer_policy(self.simulator.get_observation())
                pursuer_actions[pursuer_id] = pursuer_action
        return pursuer_actions

    def _select_evader_actions(self):
        evader_actions = {}
        for evader_id, evader_policy in self.evader_policy_dict.items():
            if evader_policy is not None:
                evader_action = evader_policy(self.simulator.get_observation())
                evader_actions[evader_id] = evader_action
        return evader_actions

@dataclass
class EnvironmentObservation:
    graph: nx.Graph
    evader_positions: dict
    pursuer_positions: dict

class PursuitGraphEvaderEnv(gym.Env):
    def __init__(self,
        simulator: PursuitGraphSimulator,
        evader_id,
        evader_policy_dict: dict,
        pursuer_policy_dict: dict,
        goal_position: Optional[Tuple],
        decoy_position: Optional[Tuple],
        path_bonus_function: Optional[PathBonusFunction],
        timeout_steps: int,
        timeout_penalty: float,
        movement_cost: float,
        deceptiveness: float,
        distance_matrix: dict,
        max_revisits: int,
        visibility: int,
        distance_metric: str,
    ):
        self.simulator = simulator
        self.evader_id = evader_id
        self.evader_policy_dict = evader_policy_dict
        self.pursuer_policy_dict = pursuer_policy_dict
        self.goal_position = goal_position
        self.decoy_position = decoy_position
        self.timeout_steps = timeout_steps
        self.timeout_penalty = timeout_penalty
        self.movement_cost = movement_cost
        self.deceptiveness = deceptiveness
        self.distance_matrix = distance_matrix
        self.max_revisits = max_revisits
        self.visibility = visibility
        self.distance_metric = distance_metric

        self.graph = simulator.graph
        self.num_evaders = len(simulator.evader_positions)
        self.num_pursuers = len(simulator.pursuer_positions)
        self.path_bonus_function = path_bonus_function
        self.path = []
        self.closest_decoy_distance = nx.shortest_path_length(self.graph, self.simulator.evader_positions[self.evader_id], self.decoy_position)

    def reset(self):
        self.path = []
        self.simulator.reset()
        self.path.append(self.simulator.evader_positions[self.evader_id])
        self.closest_decoy_distance = nx.shortest_path_length(self.graph, self.simulator.evader_positions[self.evader_id], self.decoy_position)
        return self._get_observation()

    def _step_helper(self, action, reward_dict):
        """
        A helper function that can return for the various reasons the episode may stop.
        Written as a separate function to help with control flow.
        Mutates reward_dict.
        """
        if len(self.path) > self.timeout_steps:
            reward_dict['timeout_penalty'] = self.timeout_penalty
            return True

        evader_actions = self._select_evader_actions()
        evader_actions[self.evader_id] = action
        caught_evaders = self.simulator.step_evaders(evader_actions)

        reward_dict['movement_cost'] = self.movement_cost

        if self.evader_id in caught_evaders:
            reward_dict['catch_penalty'] = -1
            return True
        
        num_revisits = self.path.count(self.simulator.evader_positions[self.evader_id])
        if num_revisits > 0:
            if num_revisits > self.max_revisits:
                reward_dict['revisit_penalty'] = -1
                return True
        
        new_pos = self.simulator.evader_positions[self.evader_id]
        if self.path_bonus_function is not None:
            # do not return True here, because deceptive bonus is additive
            reward_dict['path_bonus'] = self.path_bonus_function(self.path)

        self.path.append(new_pos)

        if self.goal_position is not None and new_pos == self.goal_position:
            reward_dict['goal_bonus'] = 1
            return True

        pursuer_actions = self._select_pursuer_actions()
        caught_evaders.update(self.simulator.step_pursuers(pursuer_actions))
        if self.evader_id in caught_evaders:
            reward_dict['catch_penalty'] = -1
            return True
        
        return False

    def step(self, action):
        reward_dict = {
            "goal_bonus": 0,
            "path_bonus": 0,
            "catch_penalty": 0,
            "timeout_penalty": 0,
            "movement_cost": 0,
        }

        done = self._step_helper(action, reward_dict)

        obs = self._get_observation()
        reward = sum(reward_dict.values())
        info = {**reward_dict}
        if self.goal_position is not None:
            info['distance_to_goal'] = nx.shortest_path_length(self.graph, self.path[-1], self.goal_position)

        return obs, reward, done, info
    
    @property
    def remaining_time(self):
        return self.timeout_steps - len(self.path)

    def _get_observation(self) -> EnvironmentObservation:
        if self.visibility != -1:
            graph = nx.ego_graph(self.graph, self.path[-1], self.visibility).copy()
        else:
            graph = self.graph.copy()
        add_position_labels_to_networkx_graph(graph, "has_evader", list(self.simulator.evader_positions.values()))
        add_position_labels_to_networkx_graph(graph, "has_pursuer", list(self.simulator.pursuer_positions.values()))
        add_position_labels_to_networkx_graph(graph, "evader_visited", self.path)
        if self.distance_metric.startswith('shortest_path'):
            if ':' in self.distance_metric:
                noise = float(self.distance_metric.split(':')[1])
            else:
                noise = 0

            add_shortest_path_distance_labels_to_networkx_graph(graph, "decoy_distance", [self.decoy_position], self.distance_matrix, noise)
            add_shortest_path_distance_labels_to_networkx_graph(graph, "goal_distance", [self.goal_position], self.distance_matrix, noise)
            self.closest_decoy_distance = min(self.closest_decoy_distance, nx.shortest_path_length(self.graph, self.path[-1], self.decoy_position))
        elif self.distance_metric == 'manhattan':
            add_manhattan_distance_labels_to_networkx_graph(graph, "decoy_distance", [self.decoy_position])
            add_manhattan_distance_labels_to_networkx_graph(graph, "goal_distance", [self.goal_position])
            self.closest_decoy_distance = min(self.closest_decoy_distance, manhattan_distance(self.path[-1], self.decoy_position))
        else:
            raise ValueError(f"Invalid `self.distance_metric`: {self.distance_metric}. Expected `shortest_path` or `manhattan`.")

        add_uniform_attribute(graph, "closest_decoy_distance", self.closest_decoy_distance)
        add_uniform_attribute(graph, "remaining_time", self.remaining_time)
        add_uniform_attribute(graph, "movement_cost", self.movement_cost)
        add_uniform_attribute(graph, "deceptiveness", float(self.deceptiveness))
        return EnvironmentObservation(
            graph=graph,
            evader_positions={**self.simulator.evader_positions},
            pursuer_positions={**self.simulator.pursuer_positions},
        )

    def _select_evader_actions(self):
        evader_actions = {}
        for evader_id, evader_policy in self.evader_policy_dict.items():
            obs = self._get_observation()
            evader_action = evader_policy(obs.graph, obs.evader_positions, obs.pursuer_positions)
            evader_actions[evader_id] = evader_action
        return evader_actions

    def _select_pursuer_actions(self):
        pursuer_actions = {}
        for pursuer_id, pursuer_policy in self.pursuer_policy_dict.items():
            obs = self._get_observation()
            pursuer_action = pursuer_policy(obs.graph, obs.evader_positions, obs.pursuer_positions)
            pursuer_actions[pursuer_id] = pursuer_action
        return pursuer_actions

def create_environment(
    graph: nx.Graph,
    value_function_table: dict,
    distance_matrix: dict,
    evader_positions: dict,
    goal_position,
    decoy_goal_position,
    movement_cost: float,
    deceptiveness: float,
    deception_type: str,
    differentiate_deception: bool,
    deception_gamma: float,
    visibility: int,
    distance_metric: str,
    timeout_steps: int,
) -> PursuitGraphEvaderEnv:
    simulator = PursuitGraphSimulator(graph, evader_positions, {})

    if deception_type == 'none':
        path_bonus_function = None
    else:
        if deception_type == 'exaggeration':
            path_bonus_function = ExaggerationDeceptionFunction(
                value_function_table,
                goal_probability_dict={goal_position: 0.5, decoy_goal_position: 0.5},
                target_goal=goal_position,
                gamma=deception_gamma,
                # graph=graph,
            )
        elif deception_type == 'ambiguity':
            path_bonus_function = AmbiguousDeceptionFunction(
                value_function_table,
                goal_probability_dict={goal_position: 0.5, decoy_goal_position: 0.5},
                target_goal=goal_position,
                gamma=deception_gamma,
                graph=graph,
            )
        elif deception_type == 'minimize_likelihood':
            path_bonus_function = MinimizeLikelihoodDeceptionFunction(
                value_function_table,
                goal_probability_dict={goal_position: 0.5, decoy_goal_position: 0.5},
                target_goal=goal_position,
                gamma=deception_gamma,
            )
        elif deception_type == 'reward_moving_to_goal':
            path_bonus_function = RewardMovingToGoalNondeceptiveFunction(
                distance_matrix=distance_matrix,
                target_goal=goal_position,
            )
        else:
            raise ValueError(f"Unexpected deception type: {deception_type}")
        path_bonus_function = path_bonus_function * deceptiveness
        if differentiate_deception:
            path_bonus_function = Differentiate(path_bonus_function)

    env = PursuitGraphEvaderEnv(
        simulator=simulator,
        evader_id=0,
        evader_policy_dict={},
        pursuer_policy_dict={},
        goal_position=goal_position,
        decoy_position=decoy_goal_position,
        path_bonus_function=path_bonus_function,
        # timeout_steps=int(4 * nx.shortest_path_length(graph, next(iter(evader_positions.values())), goal_position)),
        timeout_steps=timeout_steps,
        timeout_penalty=-0.5,
        movement_cost=movement_cost,
        deceptiveness=deceptiveness,
        distance_matrix=distance_matrix,
        max_revisits=200,
        visibility=visibility,
        distance_metric=distance_metric,
    )

    return env

def create_environment_from_scenario(scenario: 'Scenario', deception_type: str, differentiate_deception: bool, deception_gamma: float, visibility: int, distance_metric: str):
    return create_environment(
        scenario.landscape.graph,
        scenario.landscape.value_function_table,
        goal_position=scenario.goal_position,
        decoy_goal_position=scenario.decoy_goal_position,
        evader_positions={0: scenario.start_position},
        movement_cost=scenario.movement_cost,
        deceptiveness=scenario.deceptiveness,
        deception_type=deception_type,
        distance_matrix=scenario.landscape.distance_matrix,
        differentiate_deception=differentiate_deception,
        deception_gamma=deception_gamma,
        visibility=visibility,
        timeout_steps=scenario.timeout_steps,
        distance_metric=distance_metric,
    )
