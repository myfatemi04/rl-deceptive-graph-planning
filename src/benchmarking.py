import copy
from typing import Dict, List

from graph_rl import Episode, sample_episode
from plotting import (generate_heatmaps, generate_static_renders,
                      plot_episode_length_vs_statistic, plot_side_by_side)
from pursuit_evasion_simulation import create_environment_from_scenario
from scenarios import Scenario


def get_benchmark_results(scenario: Scenario, policy, deception_levels, deception_type, differentiate_deception, deception_gamma, trials_per_level, visibility, distance_metric):
    assert scenario.deceptiveness is None, "Scenario must have a blank deceptiveness level."

    episodes_by_key: Dict[float, List[Episode]] = {}

    for deception_level in deception_levels:
        episodes_by_key[deception_level] = []
        for i in range(trials_per_level):
            scenario_ = copy.copy(scenario)
            scenario_.deceptiveness = deception_level

            env = create_environment_from_scenario(
                scenario_,
                deception_type,
                differentiate_deception,
                deception_gamma,
                visibility,
                distance_metric=distance_metric,
            )
            steps = sample_episode(env, policy)
            episode = Episode(
                steps=steps,
                scenario=scenario_,
            )
            episodes_by_key[deception_level].append(episode)

    return episodes_by_key

def render_benchmark_results(scenario: Scenario, episodes_by_key: Dict[float, List[Episode]], key_name: str, out_dir: str, step: int):
    landscape = scenario.landscape
    successful_episodes_by_key = {
        key: [episode for episode in episodes if is_episode_successful(episode)]
        for key, episodes in episodes_by_key.items()
    }
    successful_episodes_by_key = {
        k: v for k, v in successful_episodes_by_key.items() if len(v) > 0
    }
    if len(successful_episodes_by_key) > 0:
        renders, titles = generate_static_renders(successful_episodes_by_key, key_name, landscape.graph, landscape.layout)
        plot_side_by_side(renders, titles, f"{out_dir}/static_renders_{scenario.name}_{step}.png")
        plot_episode_length_vs_statistic(successful_episodes_by_key, key_name, f"{out_dir}/episode_lengths_{scenario.name}_{step}.png")
    
        successful_heatmaps, titles = generate_heatmaps(successful_episodes_by_key, key_name, landscape.graph, landscape.layout)
        plot_side_by_side(successful_heatmaps, titles, f"{out_dir}/successful_heatmaps_{scenario.name}_{step}.png")

    heatmaps, titles = generate_heatmaps(episodes_by_key, key_name, landscape.graph, landscape.layout)
    plot_side_by_side(heatmaps, titles, f"{out_dir}/heatmaps_{scenario.name}_{step}.png")

def is_episode_successful(episode: Episode):
    return episode.steps[-1].info['goal_bonus'] > 0
