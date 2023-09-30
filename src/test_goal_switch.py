import time

import matplotlib.cm as cm
import networkx
import torch
from matplotlib import pyplot as plt

import graph_rendering
import landscape
import pursuit_evasion_nn_agents
from pursuit_evasion_simulation import (PursuitGraphEvaderEnv,
                                        create_environment_from_scenario)
from scenarios import get_built_in_scenarios

# land = landscape.load_landscape("gridworlds/large/MediumAmbiguity.tmx", vf_update_epsilon=-1)
# biases = [0, -50, -100, -150]

biases = [0, -100, -200, -300]
# biases = [0, -100, -175, -250]
# land = landscape.load_landscape("gridworlds/large/LargeGraph.tmx", vf_update_epsilon=-1)
land = landscape.load_landscape("gridworlds/large/LargeGraph.tmx", vf_update_epsilon=-1)
graph = land.graph
current_position = land.start_positions[0] # (4, 7)

switch_step = 100

# goal_node = land.available_goals[0]
# decoy_node = land.available_goals[1]
goal_node = land.available_goals[1]
decoy_node = land.available_goals[0]
goal_distances = networkx.single_source_shortest_path_length(graph, goal_node)
decoy_distances = networkx.single_source_shortest_path_length(graph, decoy_node)

# plot distances scatterplot
distances_to_goal = [
    goal_distances[location]
    for location in sorted(land.graph.nodes)
]
distances_to_decoy = [
    decoy_distances[location]
    for location in sorted(land.graph.nodes)
]

pos = graph_rendering.normalize_pos_dict_square(land.layout, 2400)
pos = {node: tuple(int(x) for x in pos[node]) for node in pos}

print("Initializing model...")

models = [
    {'visibility': 2, 'path': 'models/v12_different_layer_counts/k_2/2023-07-20_12-40-17/model.pt', 'deception_type': 'exaggeration'},
    {'visibility': 4, 'path': 'models/v12_different_layer_counts/k_4/2023-07-20_13-49-14/model.pt', 'deception_type': 'exaggeration'},
    {'visibility': 2, 'path': 'models/v14.2_potential/2023-07-20_18-03-26/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 2, 'path': 'models/v14.4_non_potential_unscaled/k_2/2023-07-20_23-48-38/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 4, 'path': 'models/v14.6_potential_ev/k_4/2023-07-21_09-22-14/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 4, 'path': 'models/v14.7_simpler_ambiguity/2023-07-21_15-21-25/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 2, 'path': 'models/v15.1_transient_rewards/k_2/2023-07-21_23-51-58/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 4, 'path': 'models/v23_movement_penalty/sage/exaggeration/k=4_ffl=4/2023-07-27_23-33-19/model.pt', 'deception_type': 'exaggeration'},
    {'visibility': 2, 'path': 'models/v25_varying_timeout_steps/sage/exaggeration/noise=0.0_k=2_ffl=2/2023-07-31_08-50-51/model.pt', 'deception_type': 'exaggeration'},
    {'visibility': 2, 'path': 'models/v25_varying_timeout_steps/sage/ambiguity/noise=0.0_k=2_ffl=2/2023-07-31_08-50-51/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 4, 'path': 'models/v25_varying_timeout_steps/sage/exaggeration/noise=0.0_k=4_ffl=4/2023-07-31_12-19-32/model.pt', 'deception_type': 'exaggeration'},
    {'visibility': 4, 'path': 'models/v25_varying_timeout_steps/sage/ambiguity/noise=0.0_k=4_ffl=4/2023-07-31_12-19-32/model.pt', 'deception_type': 'ambiguity'},
    {
        'visibility': 4,
        'path': "models/v25.1_varying_timeout_steps_prioritized/sage/ambiguity/distance_metric='shortest_path:0.0'_graph_depth=4_feed_forward_layers=4/2023-08-01_09-15-13/model.pt",
        'deception_type': 'ambiguity'
    },
]

checkpoint = models[-1]

visibility = checkpoint['visibility']
model = pursuit_evasion_nn_agents.PursuitGraphModel(
    dim=64,
    layers=visibility,
    keys=[
        # 'has_evader',
        # 'has_pursuer',
        'evader_visited',
        'goal_distance',
        'decoy_distance',
        'remaining_time',
        # 'deceptiveness',
        # 'closest_decoy_distance',
    ],
    augment=False,
)
print("Loading state dict...")
# Two-layer exaggeration models
# model.load_state_dict(torch.load("models/v12_different_layer_counts/k_2/2023-07-20_12-40-17/model.pt"))
# model.load_state_dict(torch.load("models/v12_different_layer_counts/k_4/2023-07-20_13-49-14/model.pt"))
model.load_state_dict(torch.load(checkpoint['path']))
model.eval()
print("Model fully loaded.")

node_size = 10
edge_thickness = 1
outline_thickness = 1

custom_node_colors = {
    goal_node: (40, 240, 80),
    decoy_node: (240, 140, 40),
    current_position: (40, 40, 240),
}
custom_node_colors_orig = {**custom_node_colors}

render = graph_rendering.render_graph_v2(land.graph, land.layout, 2400, custom_node_colors, (120, 120, 120), (50, 50, 50), node_size, edge_thickness, outline_thickness=3)
plt.imshow(render)
plt.title("Blank graph")
plt.show()

scenario, scenario_alt = get_built_in_scenarios(land, deceptiveness=1, movement_cost=0, deceptiveness_fraction=1)

print("Graph size:", len(graph.nodes))

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(wspace=0.0, hspace=0.0)
axes = axes.flatten()

d2g = []
d2d = []
paths = []

cmap = cm.get_cmap("tab10")

def sample_with_switch(env: PursuitGraphEvaderEnv, model):
    from graph_rl import Step, model_as_policy

    steps = []

    policy = model_as_policy(model, greedy=True)

    obs = env.reset()
    done = False
    while not done:
        action = policy(obs.graph, obs.evader_positions, obs.pursuer_positions)
        next_obs, reward, done, info = env.step(action)

        steps.append(Step(obs, action, reward, info))
        obs = next_obs

        if len(steps) == switch_step:
            current_position = action
            scenario_alt.start_position = current_position
            env2 = create_environment_from_scenario(
                scenario_alt,
                deception_type='none',
                differentiate_deception=False,
                deception_gamma=0.95,
                visibility=visibility,
                distance_metric='shortest_path'
            )
            env2.timeout_steps = env.remaining_time
            env = env2
            obs = env.reset()
            # set time limit

    return steps

switch_nodes = []
for i, remaining_time_bias in enumerate(biases):
    env = create_environment_from_scenario(
        scenario,
        deception_type='none',
        differentiate_deception=False,
        deception_gamma=0.95,
        visibility=visibility,
        distance_metric='shortest_path'
    )
    env.reset()

    print("Remaining time bias:", remaining_time_bias)

    success = False
    generated_path = []
    max_attempts = 5
    attempts = 0
    distances_to_goal = []
    distances_to_decoy = []
    while not success:
        attempts += 1
        if attempts > max_attempts:
            print("Max attempts reached. Showing unsuccessful run instead.")
            break
        print("Trying")
        inference_start = time.time()
        generated_path = []
        distances_to_goal = []
        distances_to_decoy = []
        steps = sample_with_switch(env, model)
        for step in steps:
            generated_path.append(step.obs.evader_positions[0])
            distances_to_goal.append(networkx.shortest_path_length(graph, step.obs.evader_positions[0], goal_node))
            distances_to_decoy.append(networkx.shortest_path_length(graph, step.obs.evader_positions[0], decoy_node))
        generated_path.append(steps[-1].action)
        # mark the switch step
        success = steps[-1].info['goal_bonus'] > 0
        inference_end = time.time()
        print("Success?", success)
        print(f"Speed: {inference_end - inference_start:.2f}s; {len(steps)/(inference_end - inference_start):.2f} nodes/s")
    shortest_path = networkx.shortest_path(graph, current_position, goal_node)
    print(f"Efficiency: {len(shortest_path)} / {len(generated_path)} = {len(shortest_path) / len(generated_path):.3f}")
    unique_nodes = set(generated_path)
    print("Number of unique nodes:", len(unique_nodes))
    arrows_drawn = set()

    render = graph_rendering.render_path_v2(land.graph, land.layout, 2400, generated_path, (255, 255, 255), {**custom_node_colors_orig, generated_path[switch_step]: (0, 0, 0)}, node_size, edge_thickness, outline_thickness)

    axes[i].set_title("Remaining time bias: " + str(remaining_time_bias))
    axes[i].imshow(render)
    axes[i].axis('off')

    for node in generated_path:
        custom_node_colors[node] = tuple(int(x * 255) for x in cmap(i)[:3])
    # custom_node_colors[generated_path[switch_step]] = (0, 0, 0)
    switch_nodes.append(generated_path[switch_step])

    d2g.append(distances_to_goal)
    d2d.append(distances_to_decoy)
    paths.append(generated_path)
plt.tight_layout(pad=0.1)
plt.show()

for switch_node in switch_nodes:
    custom_node_colors[switch_node] = (0, 0, 0)

import matplotlib.patches as mpatches

render = graph_rendering.render_graph_v2(land.graph, land.layout, 2400, custom_node_colors, (120, 120, 120), (50, 50, 50), node_size, edge_thickness, outline_thickness=2)
plt.imshow(render)
plt.title("Graph with paths")
plt.legend(handles=[
    # handle for each time bias
    mpatches.Patch(color=cmap(i), label=f"Deceptiveness level {4 - i}") # type: ignore
    for i, bias in enumerate(biases)
])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
axes = axes.flatten()

max_len = max(len(path) for path in paths)
max_dist = [max(max(d2g[i]), max(d2d[i])) for i in range(len(biases))]

for i, remaining_time_bias in enumerate(biases):
    axes[i].plot(d2g[i], label="Goal")
    axes[i].plot(d2d[i], label="Decoy")
    axes[i].set_xlim(0, max_len + 5)
    axes[i].set_ylim(0, max(max_dist) + 5)
    axes[i].legend()
    axes[i].set_ylabel("Distance")
    axes[i].set_xlabel("Time")
    axes[i].set_title(f"Remaining time bias: {remaining_time_bias}")

plt.tight_layout(pad=0.2)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
axes = axes.flatten()

# Graph projections

# plot distances scatterplot
distances_to_goal = [
    goal_distances[location]
    for location in sorted(land.graph.nodes)
]
distances_to_decoy = [
    decoy_distances[location]
    for location in sorted(land.graph.nodes)
]

for ax, path, bias in zip(axes, paths, biases):
    ax.set_title(f"Remaining time bias: {bias}")
    ax.scatter(distances_to_goal, distances_to_decoy, s=1, alpha=0.5)
    ax.scatter([goal_distances[current_position]], [decoy_distances[current_position]], s=10, c='blue', label='Start')
    ax.scatter([goal_distances[goal_node]], [decoy_distances[goal_node]], s=10, c='green', label='Goal')
    ax.scatter([goal_distances[decoy_node]], [decoy_distances[decoy_node]], s=10, c='red', label='Decoy')
    plot_x = []
    plot_y = []
    for i in range(len(path) - 1):
        plot_x.append(
            goal_distances[path[i]]
        )
        plot_y.append(
            decoy_distances[path[i]]
        )
    ax.plot(plot_x, plot_y, c='black', alpha=0.5, label='Path')
    ax.legend()
    ax.set_xlabel("Distance to goal")
    ax.set_ylabel("Distance to decoy")

plt.show()

