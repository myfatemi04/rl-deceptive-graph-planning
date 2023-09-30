import time

import networkx

import graph_rl
from pursuit_evasion_simulation import create_environment_from_scenario


def simplify_path(path, simplification_window=5):
    # if a node is encountered again that was encountered less than five steps ago, remove all nodes in between
    changed = True
    while changed:
        changed = False
        for i in range(len(path)):
            if i < simplification_window:
                continue
            if path[i] in path[i - simplification_window:i]:
                pos = (i - simplification_window) + path[i - simplification_window:i].index(path[i])
                path = path[:pos] + path[i:]
                changed = True
                break
    return path

def create_env(land, start_position, goal_node, decoy_node, visibility, timeout_steps):
    scenario = graph_rl.Scenario(
        name='Rendering',
        landscape=land,
        start_position=start_position,
        goal_position=goal_node,
        decoy_goal_position=decoy_node,
        deceptiveness=1,
        movement_cost=0,
        timeout_steps=timeout_steps,
    )
    env = create_environment_from_scenario(
        scenario,
        deception_type='none',
        differentiate_deception=False,
        deception_gamma=0.95,
        visibility=visibility,
        distance_metric='shortest_path'
    )
    return env

def simulate_environment(model, land, start_position, goal_node, decoy_node, visibility, timeout_steps, greedy=True):
    env = create_env(land, start_position, goal_node, decoy_node, visibility, timeout_steps)
    env.reset()

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
        steps = graph_rl.sample_episode(env, graph_rl.model_as_policy(model, greedy=greedy))
        for step in steps:
            generated_path.append(step.obs.evader_positions[0])
            distances_to_goal.append(land.distance_matrix[goal_node][step.obs.evader_positions[0]])
            distances_to_decoy.append(land.distance_matrix[decoy_node][step.obs.evader_positions[0]])
        generated_path.append(steps[-1].action)
        success = steps[-1].info['goal_bonus'] > 0
        inference_end = time.time()
        print("Success?", success)
        print(f"Speed: {inference_end - inference_start:.2f}s; {len(steps)/(inference_end - inference_start):.2f} nodes/s")
    simplified_path = simplify_path(generated_path)
    shortest_path = networkx.shortest_path(land.graph, start_position, goal_node)
    unique_nodes = set(generated_path)
    
    print("Path length:", len(generated_path))
    print("Simplified path length:", len(simplified_path))
    print(f"Efficiency: {len(shortest_path)} / {len(generated_path)} = {len(shortest_path) / len(generated_path):.3f}")
    print("Number of unique nodes:", len(unique_nodes))

    return {
        "path": {
            "generated": generated_path,
            "simplified": simplified_path,
        },
        "distances": {
            "goal": distances_to_goal,
            "decoy": distances_to_decoy,
        },
        "success": success,
        "time": inference_end - inference_start, # type: ignore
    }
