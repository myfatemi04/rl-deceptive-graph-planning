import random
from dataclasses import dataclass
from typing import List

import numpy as np

from landscape import Landscape


@dataclass
class Scenario:
    name: str
    landscape: Landscape
    start_position: tuple
    goal_position: tuple
    decoy_goal_position: tuple
    deceptiveness: float
    movement_cost: float
    timeout_steps: int

def get_built_in_scenarios(landscape: Landscape, deceptiveness, movement_cost: float, deceptiveness_fraction: float):
    available_goals = list(landscape.available_goals)
    start_positions = list(landscape.start_positions)
    assert len(start_positions) == 1, "Only one start position is supported for now"
    scenarios: List[Scenario] = []
    for goal_idx in range(len(available_goals)):
        for decoy_goal_idx in range(len(available_goals)):
            if goal_idx == decoy_goal_idx:
                continue

            start_position = start_positions[0]
            goal_position = available_goals[goal_idx]
            decoy_goal_position = available_goals[decoy_goal_idx]

            a, b = get_min_and_max_exaggeration_episode_durations(
                landscape,
                start_position,
                goal_position,
                decoy_goal_position
            )
            timeout_steps = round(a + (b - a) * deceptiveness_fraction)

            scenario = Scenario(
                name='{}-{}-{}'.format(landscape.name, available_goals[goal_idx], available_goals[decoy_goal_idx]),
                landscape=landscape,
                start_position=start_position,
                goal_position=goal_position,
                decoy_goal_position=decoy_goal_position,
                deceptiveness=deceptiveness,
                movement_cost=movement_cost,
                timeout_steps=timeout_steps,
            )

            scenarios.append(scenario)
            
    return scenarios

def get_min_and_max_exaggeration_episode_durations(landscape, start_position, goal_position, decoy_goal_position):
    distance_to_goal = landscape.distance_matrix[goal_position][start_position]
    distance_to_decoy = landscape.distance_matrix[decoy_goal_position][start_position]
    distance_from_decoy_to_goal = landscape.distance_matrix[goal_position][decoy_goal_position]
    time_required_for_full_deceptiveness = distance_to_decoy + distance_from_decoy_to_goal
    time_required_for_no_deceptiveness = distance_to_goal

    return (time_required_for_no_deceptiveness, time_required_for_full_deceptiveness)

def get_random_scenario(landscape: Landscape, deceptiveness: float, movement_cost: float):
    nodes = list(landscape.graph.nodes)
    start_idx, goal_idx, decoy_goal_idx = np.random.choice(len(nodes), size=3, replace=False)
    # randomly select an amount of remaining time
    timeout_steps = round(random.uniform(*get_min_and_max_exaggeration_episode_durations(
        landscape,
        nodes[start_idx],
        nodes[goal_idx],
        nodes[decoy_goal_idx]
    )))
    return Scenario(
        name='{}-{}-{}'.format(landscape.name, nodes[goal_idx], nodes[decoy_goal_idx]),
        landscape=landscape,
        start_position=nodes[start_idx],
        goal_position=nodes[goal_idx],
        decoy_goal_position=nodes[decoy_goal_idx],
        deceptiveness=deceptiveness,
        movement_cost=movement_cost,
        timeout_steps=timeout_steps,
    )
