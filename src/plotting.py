from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm

from graph_rendering import draw_path, render_graph
from graph_rl import Episode

_disable_plot_windows = False

def disable_plot_windows():
    global _disable_plot_windows
    _disable_plot_windows = True
    matplotlib.use('Agg')

def plt_show():
    if _disable_plot_windows:
        plt.close()
    else:
        plt.show()

def roll(history, window=20):
    return np.convolve(np.array(history), np.ones(window) / window, mode='valid')

def slug(string):
    return '_'.join([token.lower() for token in string.split(" ")])

def plot_rolling_stat(history, stat_name, base_dir, rolling_window=20):
    rolling = roll(history, rolling_window)
    plt.title(stat_name)
    plt.plot(rolling)
    slug = '_'.join([token.lower() for token in stat_name.split(" ")])
    plt.savefig(f"{base_dir}/pe_{slug}.png")
    plt_show()

def plot_episodes(episodes_to_watch, base_dir, episode_number, test_landscapes, model, select_movement_cost, eval_discrete_levels, deception_type):
    path_bonuses = np.array([sum(s.info['path_bonus'] for s in e.steps) for e in episodes_to_watch])
    episode_lengths = np.array([len(e.steps) for e in episodes_to_watch])
    goal_reached = np.array([e.steps[-1].info['goal_bonus'] > 0 for e in episodes_to_watch])
    
    plt.scatter(episode_lengths, path_bonuses, c=goal_reached, alpha=0.1)
    plt.xlabel("Episode length")
    plt.ylabel("Path bonus")
    plt.savefig(f"{base_dir}/episode_length_vs_path_bonus_{episode_number}.png")
    plt_show()

    ax1: plt.Axes = plt.gca()
    ax2: plt.Axes = ax1.twinx()
    ax1.set_ylabel("Goal rate")
    ax1.plot(roll(goal_reached, 20), color='orange', label='Goal reached rate')
    ax2.set_ylabel("Path bonus")
    ax2.plot(roll(path_bonuses, 20), color='blue', label='Path bonus')
    plt.legend()
    plt.title("Goal rate vs. Path bonus")
    plt.savefig(f"{base_dir}/combined_plot_{episode_number}.png")
    plt_show()

def render_static_episode(episode: Episode, graph, layout: dict):
    """
    Render the most recent episode in BGR format.
    """
    path = [next(iter(step.obs.evader_positions.values())) for step in episode.steps] + [episode.steps[-1].action]
    static_render = draw_path(graph, layout, path, node_size=10, edge_width=3, custom_node_colors={
        # Start of path: blue
        path[0]: (255, 50, 50),
        # End of path: red
        path[-1]: (50, 50, 255),
        # Goal: green
        episode.scenario.goal_position: (50, 255, 50),
        # Decoy goal: orange
        episode.scenario.decoy_goal_position: (50, 127, 255),
    })
    return static_render

def render_heatmap(episodes: List[Episode], graph, layout: dict):
    cmap = cm.get_cmap('plasma')

    visit_counts = {}
    max_visit_count = 0
    for episode in episodes:
        seen_positions = set()
        for step in episode.steps:
            position = next(iter(step.obs.evader_positions.values()))
            if position in seen_positions:
                continue
            seen_positions.add(position)
            visit_counts[position] = visit_counts.get(position, 0) + 1
            max_visit_count = max(max_visit_count, visit_counts[position])
    node_colors = {
        node: tuple([int(255 * value) for value in cmap(visit_counts[node] / max_visit_count)[:3]])[::-1] for node in visit_counts
    }
    image = render_graph(
        graph,
        layout,
        node_size=10,
        edge_width=3,
        node_colors={
            **node_colors,
            episodes[-1].scenario.goal_position: (50, 255, 50),
            episodes[-1].scenario.decoy_goal_position: (50, 127, 255),
        }
    )
    return image


def generate_static_renders(
    successful_episodes_by_key,
    key_name,
    graph,
    layout,
):
    n_cols = len(successful_episodes_by_key)

    assert n_cols > 0, "Must have at least one successful episode"

    order = sorted(successful_episodes_by_key.keys())
    static_renders = [render_static_episode(successful_episodes_by_key[key][-1], graph, layout) for key in order]
    titles = [f"{key_name} = {key}" for key in order]

    return static_renders, titles

def generate_heatmaps(episodes_by_level: dict, key_name: str, graph, layout: dict):
    titles = [f"{key_name} = {key}" for key in episodes_by_level.keys()]
    heatmaps = [render_heatmap(episodes, graph, layout) for episodes in episodes_by_level.values()]
    return heatmaps, titles

def plot_side_by_side(images, titles, out_path):
    n_cols = len(images)
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    if type(axs) != np.ndarray:
        axs = [axs]

    for ax, image, title in zip(axs, images, titles): # type: ignore
        ax: plt.Axes
        ax.set_title(title)
        ax.imshow(image[:, :, ::-1])
        ax.axis('off')
    
    plt.savefig(out_path)
    plt_show()

def plot_episode_length_vs_statistic(successful_episodes_by_key: dict, key_name: str, out_file: str):
    x = []
    y = []
    for key, episodes in sorted(successful_episodes_by_key.items()):
        x.extend([key] * len(episodes))
        y.extend([len(episode.steps) for episode in episodes])

    sns.set_style("whitegrid")
    plt.title(f"{key_name} vs. Offline episode length")
    sns.boxenplot(x=x, y=y)
    plt.xlabel(key_name)
    plt.ylabel("Offline episode length")
    plt.savefig(out_file)
    plt_show()