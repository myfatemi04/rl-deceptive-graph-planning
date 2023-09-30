import os
from dataclasses import dataclass
from typing import List

import networkx as nx

from generate_assumed_value_function import calculate_value_function_dict
from grid_world import load_grid


class CachedDistanceMatrix(dict):
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    def __missing__(self, key):
        self[key] = dict(nx.single_source_shortest_path_length(self.graph, key))
        return self[key]

@dataclass
class Landscape:
    graph: nx.Graph
    layout: dict
    name: str
    available_goals: list
    start_positions: list
    value_function_table: dict
    distance_matrix: CachedDistanceMatrix

def load_landscape(file, vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.99, vf_update_epsilon=0.001, generate_all_goals=True) -> Landscape:
    name = file.split('/')[-1][:-4]
    grid = load_grid(file)
    vf_table = calculate_value_function_dict(
        graph=grid.graph,
        # Calculate value function for all nodes
        goal_nodes=list(grid.graph.nodes) if generate_all_goals else grid.goal_positions,
        transition_reward=vf_movement_penalty,
        discount_factor=vf_discount_factor,
        goal_initial_value=vf_goal_initial_value,
        update_epsilon=vf_update_epsilon,
    ) if vf_update_epsilon > 0 else None
    landscape = Landscape(
        graph=grid.graph,
        layout=grid.graph_layout,
        name=name,
        available_goals=grid.goal_positions,
        start_positions=grid.start_positions,
        value_function_table=vf_table, # type: ignore
        distance_matrix=CachedDistanceMatrix(grid.graph)
    )
    return landscape

def load_landscapes(folder='gridworlds', vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.99) -> List[Landscape]:
    landscapes: List[Landscape] = []
    for file in os.listdir(folder):
        if file.endswith(".tmx"):
            landscapes.append(
                load_landscape(os.path.join(folder, file), vf_movement_penalty, vf_goal_initial_value, vf_discount_factor)
            )
    return landscapes
