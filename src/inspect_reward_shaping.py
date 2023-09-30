import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import deception_types as dt
import landscape
from graph_rendering import render_graph, render_graph_v2


def create_path(graph: nx.Graph, checkpoints):
    path = []
    for i in range(len(checkpoints) - 1):
        path += nx.shortest_path(graph, checkpoints[i], checkpoints[i + 1])[:-1]
    path.append(checkpoints[-1])
    return path

# scape = landscape.load_landscape('gridworlds/Maze.tmx', vf_movement_penalty=-0.05, vf_goal_initial_value=1)

# paths = [
#     create_path(scape.graph, [(11, 0), (5, 5), (0, 5)]),
#     create_path(scape.graph, [(11, 0), (2, 11), (0, 5)]),
# ]

# goals = [(2, 7), (0, 5)]
# exaggeration_function = ExaggerationDeceptionFunction(
#     scape.value_function_table,
#     {(0, 5): 0.5, (2, 7): 0.5},
#     target_goal=(0, 5),
#     gamma=0.95,
# )


def plot_inspection(ls):
    # for g in goals[:1]:
    #     # potential = scape.value_function_table[g]
    g = goals[0]
    print(goals)
    potential = {
        node: deception_fn([start_pos, node]) for node in ls.graph.nodes
    }
    vf_min = min(potential.values())
    vf_max = max(potential.values())

    # vf_render = render_graph(ls.graph, ls.layout, node_size=10, edge_width=3, node_colors={
    #     node: to_256(colormap((potential[node] - vf_min) / (vf_max - vf_min)))
    #     for node in ls.graph.nodes
    # })
    vf_render = render_graph_v2(ls.graph, ls.layout, node_size=10, edge_thickness=1, outline_thickness=0, node_colors={**{
        # (5, 0): (0, 255, 0),
        node: to_256(colormap((potential[node] - vf_min) / (vf_max - vf_min)))
        for node in ls.graph.nodes
    }, goals[0]: (0, 255, 0), goals[1]: (255, 0, 0), start_pos: (0, 0, 255)}, image_size=1200)

    print(vf_min, vf_max)

    plt.title("Goal: " + str(g))
    plt.imshow(vf_render)
    plt.show()
    
    return

    for path in paths:
        deceptive_rewards = []
        for i in range(1, len(path)):
            rw = potential[path[i]]
            deceptive_rewards.append(rw)
        deceptive_rewards = np.array(deceptive_rewards)
        movement_rewards = -0.05 * np.ones(len(path) - 1)

        for g in goals:
            print("Goal", g)
            for i in range(len(path)):
                print(f"{ls.value_function_table[g][path[i]]:.2f}", end=' ')
            print()

        plt.plot(deceptive_rewards, label="Deceptive reward")
        plt.plot(movement_rewards, label="Movement reward")
        plt.plot(deceptive_rewards + movement_rewards, label="Total reward")
        plt.legend()
        plt.show()

        deceptive_rewards = np.array(deceptive_rewards)
        rewards_norm = deceptive_rewards.copy()
        rewards_norm -= rewards_norm.min()
        rewards_norm /= rewards_norm.max()

        rendered = render_graph_v2(ls.graph, ls.layout, node_size=10, edge_thickness=3, node_colors={
            # (5, 0): (0, 255, 0),
            node: to_256(colormap(reward))
            for node, reward in zip(path[1:], rewards_norm)
        }, image_size=1200)

        print("Total reward:", sum(deceptive_rewards))
        
        plt.title("Reward over time")
        plt.plot(deceptive_rewards)
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.show()

        plt.imshow(rendered)
        plt.show()

DeceptionFunction = dt.AmbiguousDeceptionFunction

landscapes = landscape.load_landscapes(folder='gridworlds/square_rect_based', vf_movement_penalty=0, vf_goal_initial_value=1)
# landscapes = [landscape.load_landscape('gridworlds/square_rect_based/8x8B.tmx', vf_movement_penalty=0, vf_goal_initial_value=1)]
for ls in landscapes:
    # scape = landscape.load_landscape('gridworlds/AmbiguitySquare.tmx', vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.99)
    # ls = landscape.load_landscape('gridworlds/large/LargeAmbiguity.tmx', vf_update_epsilon=-1)
    # ls = landscape.load_landscape('gridworlds/large/MediumAmbiguity.tmx', vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.5 ** (1 / 45), generate_all_goals=False)
    # ls = landscape.load_landscape('gridworlds/large/LargeGraph2.tmx', vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.5 ** (1 / 150), generate_all_goals=False)
    # ls = landscape.load_landscape('gridworlds/AmbiguitySquare.tmx', vf_movement_penalty=-0.05, vf_goal_initial_value=1, vf_discount_factor=0.99, generate_all_goals=False)

    # start_pos = (8, 4)
    try:
        start_pos = ls.start_positions[0]
    except:
        print("No start positions found")
        print(ls.name)
        continue
    # start_pos = (2, 31)

    # paths = [
    #     create_path(ls.graph, [start_pos, (8, 4), (0, 4), (0, 0)]),
    #     create_path(ls.graph, [start_pos, (0, 0)]),
    # ]

    # print(ls.available_goals)

    goals = ls.available_goals[::-1]

    goal_path_length = nx.shortest_path_length(ls.graph, start_pos, goals[0])
    # deception_fn = AmbiguousDeceptionFunction(
    #     ls.value_function_table,
    #     {goal: 0.5 for goal in goals},
    #     target_goal=goals[0],
    #     # lose 50% deceptiveness every time we get 50% closer to the goal
    #     gamma=0.5 ** (1 / goal_path_length),
    #     graph=ls.graph
    # )
    deception_fn = DeceptionFunction(
        ls.value_function_table,
        {goal: 0.5 for goal in goals},
        target_goal=goals[0],
        # lose 50% deceptiveness every time we get 50% closer to the goal
        gamma=0.5 ** (1 / goal_path_length),
        graph=ls.graph
    )
    # deception_fn = ExaggerationDeceptionFunction(
    #     ls.value_function_table,
    #     {goal: 0.5 for goal in goals},
    #     target_goal=goals[0],
    #     # lose 50% deceptiveness every time we get 50% closer to the goal
    #     gamma=0.5 ** (1 / goal_path_length),
    #     # graph=ls.graph
    # )

    to_256 = lambda color: tuple([int(x * 255) for x in color[:3]])
    colormap = cm.get_cmap("viridis")

    plot_inspection(ls)

exit()

path = nx.shortest_path(ls.graph, start_pos, ls.available_goals[0]) # type: ignore
for node in path:
    distance_decoy = nx.shortest_path_length(ls.graph, node, ls.available_goals[1])
    distance_goal = nx.shortest_path_length(ls.graph, node, ls.available_goals[0])
    print(distance_decoy, distance_goal)

### Allow simulation
for ep in range(100):
    print("New Episode")
    path: list = [start_pos]
    rewards = []
    for step in range(300):
        rendered = render_graph_v2(ls.graph, ls.layout, node_size=4, edge_thickness=1, outline_thickness=0, node_colors={**{
            # (5, 0): (0, 255, 0),
            node: to_256(colormap(reward))
            for node, reward in zip(path[1:], rewards)
        }, goals[0]: (0, 255, 0), goals[1]: (255, 0, 0)}, image_size=1200)
        cv2.imshow("Scene", rendered)
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
            break
        elif key == ord('q'):
            exit()
        if path[-1] not in ls.graph.nodes:
            break
        
        distance_decoy = nx.shortest_path_length(ls.graph, path[-1], ls.available_goals[1])
        distance_goal = nx.shortest_path_length(ls.graph, path[-1], ls.available_goals[0])
        print(distance_decoy, distance_goal, ls.available_goals[0], ls.available_goals[1])
        # rewards.append(0)
        rewards.append(deception_fn(path))
        print("Reward:", rewards[-1], "Total:", sum(rewards), "Path length:", len(path))
