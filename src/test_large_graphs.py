"""
Usage: test_large_graphs.py [options]

Options:
    --show-distance-space    Show the distance space scatterplot [default: 0]
    --show-blank-graph       Show the blank graph [default: 0]

"""

doc = __doc__

import os

import networkx
from docopt import docopt
from matplotlib import pyplot as plt

import checkpoints
import graph_rendering
import landscape
from run_policy import simulate_environment

landscapes = [
    ('large/MediumAmbiguity', 32),
    ('large/LargeGraph', 100),
    ('large/LargeGraph2', 100),
    ('square_rect_based/8x8E', 8),
    ('square_rect_based/16x16E', 16),
]

def render_multiple_paths():
    landscape_file, landscape_size = landscapes[1]
    land = landscape.load_landscape(f'gridworlds/{landscape_file}.tmx', vf_update_epsilon=-1)

    specs = {
        'exaggeration': {
            'color': (255, 0, 0),
            'extra_time_coefficient': 1.25,
            'checkpoint': checkpoints.models[-1],
            'num_layers': 4,
        },
        'ambiguity': {
            'color': (127, 0, 255),
            'extra_time_coefficient': 1.0,
            'checkpoint': checkpoints.models[-3],
            'num_layers': 2,
        }
    }
    shortest_path_color = (0, 255, 255)

    out_dir = f'figures/path_samples_and_renders/overlaid/{land.name}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    graph = land.graph
    start_position = land.start_positions[0] # (4, 7)
    goal_node = land.available_goals[1]
    decoy_node = land.available_goals[0]
    goal_node, decoy_node = decoy_node, goal_node
    goal_distances = land.distance_matrix[goal_node]
    decoy_distances = land.distance_matrix[decoy_node]
    pos = graph_rendering.normalize_pos_dict_square(land.layout, 2400)
    pos = {node: tuple(int(x) for x in pos[node]) for node in pos}
    landmark_colors = {
        goal_node: (40, 240, 80),
        decoy_node: (240, 140, 40),
        start_position: (40, 40, 240),
    }

    distance_to_goal = goal_distances[start_position]
    distance_to_decoy = decoy_distances[start_position]
    distance_between_goal_and_decoy = networkx.shortest_path_length(graph, goal_node, decoy_node)
    lower = distance_to_goal
    upper = distance_to_decoy + distance_between_goal_and_decoy

    node_size = 640//landscape_size
    edge_thickness = 80//landscape_size
    outline_thickness = 80//landscape_size

    # plot distances scatterplot
    distances_to_goal = [
        goal_distances[location]
        for location in sorted(land.graph.nodes)
    ]
    distances_to_decoy = [
        decoy_distances[location]
        for location in sorted(land.graph.nodes)
    ]

    arguments = docopt(doc)
    show_distance_space = int(arguments['--show-distance-space'])

    if show_distance_space:
        plt.title("Graph projection to distance space")
        plt.scatter(distances_to_goal, distances_to_decoy, s=1, alpha=0.5)
        plt.scatter([goal_distances[start_position]], [decoy_distances[start_position]], s=10, c='blue', label='Start')
        plt.scatter([goal_distances[goal_node]], [decoy_distances[goal_node]], s=10, c='green', label='Goal')
        plt.scatter([goal_distances[decoy_node]], [decoy_distances[decoy_node]], s=10, c='red', label='Decoy')
        plt.legend()
        plt.xlabel("Distance to goal")
        plt.ylabel("Distance to decoy")
        plt.show()

    node_colors = {}
    for key in specs:
        model = checkpoints.load_checkpoint(specs[key]['checkpoint'])
        result = simulate_environment(
            model, land, start_position, goal_node, decoy_node, specs[key]['num_layers'],
            distance_to_goal + int((upper - lower) * specs[key]['extra_time_coefficient']), # type: ignore
        )
        path = result['path']['simplified']
        for item in path:
            node_colors[item] = specs[key]['color']

    for item in networkx.shortest_path(graph, start_position, goal_node):
        node_colors[item] = shortest_path_color

    render = graph_rendering.render_graph_v2(land.graph, land.layout, 2400, {**node_colors, **landmark_colors}, node_size=node_size, edge_thickness=edge_thickness, outline_thickness=outline_thickness)
    plt.imshow(render)
    # add labels
    plt.title("Graph with paths")
    # import matplotlib.patches as mp
    # plt.legend(handles=[
    #     mp.Patch(color='red', label="Exaggeration path"),
    #     mp.Patch(color='blue', label="Ambiguity path"),
    #     mp.Patch(color='green', label="Shortest path"),
    # ], loc='lower left')
    plt.show()

    # # Render video of path
    # for i in range(4):
    #     graph_rendering.render_video(land.graph, land.layout, 1200, paths[i], 4, landmark_colors, f"path_{i}.avi", fps=16)

    # import matplotlib.patches as mpatches

    # render = graph_rendering.render_graph_v2(land.graph, land.layout, 2400, landmark_colors, (120, 120, 120), (50, 50, 50), node_size, edge_thickness, outline_thickness=2)
    # plt.imshow(render)
    # plt.title("Graph with paths")
    # plt.legend(handles=[
    #     # handle for each time bias
    #     mpatches.Patch(color=cmap(i), label=f"Deceptiveness level {4 - i}")
    #     for i, bias in enumerate(biases)
    # ])
    # plt.show()

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # axes = axes.flatten()

    # max_len = max(len(path) for path in paths)
    # max_dist = [max(max(d2g[i]), max(d2d[i])) for i in range(len(biases))]

    # for i, extra_time in enumerate(biases):
    #     axes[i].plot(d2g[i], label="Goal")
    #     axes[i].plot(d2d[i], label="Decoy")
    #     axes[i].set_xlim(0, max_len + 5)
    #     axes[i].set_ylim(0, max(max_dist) + 5)
    #     axes[i].legend()
    #     axes[i].set_ylabel("Distance")
    #     axes[i].set_xlabel("Time")
    #     axes[i].set_title(f"Extra time: {extra_time}")

    # plt.tight_layout(pad=0.2)
    # plt.show()

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # axes = axes.flatten()

    # # Graph projections

    # # plot distances scatterplot
    # distances_to_goal = [
    #     goal_distances[location]
    #     for location in sorted(land.graph.nodes)
    # ]
    # distances_to_decoy = [
    #     decoy_distances[location]
    #     for location in sorted(land.graph.nodes)
    # ]

    # for ax, path, bias in zip(axes, paths, biases):
    #     ax.set_title(f"Remaining time bias: {bias}")
    #     ax.scatter(distances_to_goal, distances_to_decoy, s=1, alpha=0.5)
    #     ax.scatter([goal_distances[start_position]], [decoy_distances[start_position]], s=10, c='blue', label='Start')
    #     ax.scatter([goal_distances[goal_node]], [decoy_distances[goal_node]], s=10, c='green', label='Goal')
    #     ax.scatter([goal_distances[decoy_node]], [decoy_distances[decoy_node]], s=10, c='red', label='Decoy')
    #     plot_x = []
    #     plot_y = []
    #     for i in range(len(path) - 1):
    #         plot_x.append(
    #             goal_distances[path[i]]
    #         )
    #         plot_y.append(
    #             decoy_distances[path[i]]
    #         )
    #     ax.plot(plot_x, plot_y, c='black', alpha=0.5, label='Path')
    #     ax.legend()
    #     ax.set_xlabel("Distance to goal")
    #     ax.set_ylabel("Distance to decoy")

    # plt.show()

if __name__ == "__main__":
    render_multiple_paths()
