from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def _after(s, search_string):
    return s[s.index(search_string) + len(search_string):]

def _before(s, search_string):
    return s[:s.index(search_string)]

def transpose_where(where_result):
    for i in range(len(where_result[0])):
        yield tuple(where_result[j][i] for j in range(len(where_result)))

def whereT(query) -> List[Tuple[int, int]]:
    return list(transpose_where(np.where(query))) # type: ignore

def load_tilemap(path: str):
    # Our graphs are simple enough.

    with open(path, "r") as f:
        content = f.read()

    grid_text = _before(_after(content, '<data encoding="csv">'), '</data>')
    grid_lines = grid_text.strip().split("\n")
    grid_lines = [line.rstrip(',') for line in grid_lines]
    grid = [[int(x) for x in line.split(",")] for line in grid_lines]
    grid = np.array(grid)

    return grid

def in_grid(grid_shape, y, x):
    return grid_shape[0] > y >= 0 and grid_shape[1] > x >= 0

def neighbors(y, x):
    return (
        (y + 1, x),
        (y, x + 1),
        (y - 1, x),
        (y, x - 1)
    )

def grid_to_nx_graph(obstacle_mask: np.ndarray):
    empty_mask = ~obstacle_mask
    graph = nx.Graph()
    for (y, x) in zip(*np.where(empty_mask)): # type: ignore
        assert in_grid(empty_mask.shape, y, x)
        assert empty_mask[y, x]
        for y1, x1 in neighbors(y, x):
            if not in_grid(empty_mask.shape, y1, x1):
                continue
            
            if empty_mask[y1, x1]:
                graph.add_edge((y, x), (y1, x1))
    return graph

GridPoint = Tuple[int, int]

@dataclass
class ImportedGridWorld:
    tiles: np.ndarray
    obstacle_mask: np.ndarray
    goal_positions: List[GridPoint]
    start_positions: List[GridPoint]
    graph: nx.Graph
    graph_layout: Dict[GridPoint, Tuple[int, int]]

GOAL_POSITION_TILE_TYPE = 2
START_POSITION_TILE_TYPE = 3
OBSTACLE_TILE_TYPE = 4

def get_obstacle_mask(tilemap: np.ndarray):
    return tilemap == OBSTACLE_TILE_TYPE

def get_goal_positions(tilemap: np.ndarray):
    return whereT(tilemap == GOAL_POSITION_TILE_TYPE)

def get_start_positions(tilemap: np.ndarray):
    return whereT(tilemap == START_POSITION_TILE_TYPE)

def create_graph_layout(graph):
    return {
        (y, x): (x, y)
        for (y, x) in graph.nodes
    }

def load_grid(path: str):
    tilemap = load_tilemap(path)

    obstacle_mask = get_obstacle_mask(tilemap)
    graph = grid_to_nx_graph(obstacle_mask)

    return ImportedGridWorld(
        tiles=tilemap,
        obstacle_mask=obstacle_mask,
        goal_positions=get_goal_positions(tilemap),
        start_positions=get_start_positions(tilemap),
        graph=graph,
        graph_layout=create_graph_layout(graph)
    )

def main():
    from generate_assumed_value_function import render_sample

    grid = load_grid("gridworlds/TinyMap.tmx")
    render_sample(grid.graph, grid.graph_layout, [*grid.goal_positions])

if __name__ == '__main__':
    main()
