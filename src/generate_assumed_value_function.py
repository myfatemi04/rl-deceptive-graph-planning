import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from caching import pickle_cached


@pickle_cached('value_tables')
def calculate_value_function_for_goal(graph: nx.Graph, goal_node, transition_reward, discount_factor, goal_initial_value, update_epsilon=1e-3, normalize=False, verbose=False):
    """
    Calculates the expected value of being at each node, given that eventually, our agent wants to reach a certain goal node.

    Parameters:
     - `graph`: A NetworkX graph representing possible locations
     - `goal_node`: The node we assume the agent wants to reach
     - `cost_function`: A float, int, dict, or function that designates the cost of traversing from one node to another.
        If a float or int, the cost is the same for all edges.
        If a dict, the cost is the value of the dict keyed by the tuple `(start_node, end_node)`.
        If a function, the cost is the return value of the function when called with the source and destination nodes as arguments.
    - `discount_factor`: A float between 0 and 1 that represents how much the agent values future rewards. Used to calculate Q-values.
    - `goal_initial_value`: A value initially assigned to the goal node.
    - `softmax_alpha`: A float that represents the temperature of the softmax function used to calculate Q-values.
    - `update_epsilon`: A float that represents the maximum difference between the old and new value functions before the algorithm stops.
    """

    assert graph.has_node(goal_node), f"Goal node does not exist in graph: {goal_node}"

    # verbose = True

    if type(transition_reward) in [float, int]:
        value = transition_reward
        transition_reward = lambda start_node, end_node: value
    elif type(transition_reward) == dict:
        transition_reward = lambda start_node, end_node: transition_reward[start_node, end_node] # type: ignore
    else:
        assert callable(transition_reward)
    
    value_function = {}
    value_function[goal_node] = goal_initial_value

    iterate = True

    num_iterations = 0
    while iterate:
        num_iterations += 1
        total_change = 0
        min_value = np.inf
        max_value = -np.inf

        iterate = False

        new_value_function = {}
        new_value_function[goal_node] = goal_initial_value
        
        # softmax value iteration
        for node in graph.nodes:
            if node == goal_node:
                continue

            action_q_values = np.array([transition_reward(node, neighbor) + discount_factor * value_function.get(neighbor, 0) for neighbor in graph.neighbors(node)])
            # Support numerical stability. Subtracting the same value from each term going into a softmax
            norm_q_values = (action_q_values - action_q_values.mean())
            new_value = (action_q_values * (np.exp(norm_q_values) / np.exp(norm_q_values).sum())).sum()
            # new_value = softmax_alpha * np.log(np.exp(action_q_values / softmax_alpha).sum())
            # new_value = np.mean(action_q_values)

            new_value_function[node] = new_value
            update_amount = abs(new_value - value_function.get(node, 0))
            total_change += update_amount ** 2

            if update_amount > update_epsilon:
                iterate = True
        
        value_function = new_value_function

        if verbose:
            print("RMSE:", (total_change / len(graph.nodes)) ** 0.5)

    if verbose:
        print("Converged after", num_iterations, "iterations")

    if normalize:
        max_value = max(value_function.values())
        min_value = min(value_function.values())
        if verbose:
            print("Min and max values:", min_value, max_value)
        for node in graph.nodes:
            value_function[node] = value_function[node] / max_value
            # value_function[node] = (value_function[node] - min_value) / (max_value - min_value)

    return value_function

def calculate_value_function_dict(graph: nx.Graph, goal_nodes, transition_reward, discount_factor, goal_initial_value, update_epsilon=1e-3, normalize=False, verbose=False):
    return {
        node: calculate_value_function_for_goal(graph, node, transition_reward, discount_factor, goal_initial_value, update_epsilon, normalize, verbose)
        for node in goal_nodes # tqdm(goal_nodes, desc='Calculating value function...')
    }

def to_uint_cmap(cmap):
    return lambda x: tuple(int(c * 255) for c in cmap(x))

def render_sample(graph, pos=None, goal_nodes=None):
    import graph_rendering

    if pos is None:
        pos = nx.spring_layout(graph, iterations=500)
    if goal_nodes is None:
        goal_nodes = graph.nodes

    for goal_node in goal_nodes:
        for df in [0.9, 0.95, 0.999]:
            node_values = calculate_value_function_for_goal(
                graph,
                goal_node=goal_node,
                transition_reward=-1,
                discount_factor=df,
                goal_initial_value=5,
                update_epsilon=0.05,
                normalize=True,
                verbose=True,
            )
            cmap = to_uint_cmap(cm.get_cmap("viridis"))

            image = graph_rendering.render_graph(
                graph,
                pos, # type: ignore
                node_size=5,
                edge_width=1,
                node_colors={
                    node: cmap(value) if node != goal_node else (255, 255, 255)
                    for (node, value) in node_values.items()
                }
            )
            
            plt.imshow(image)
            plt.colorbar()
            plt.show()

# if __name__ == '__main__':
#     from graph_examples import CHESSBOARD_8

#     render_sample(CHESSBOARD_8)
