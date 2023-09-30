"""
Usage: continuous_sim.py [options]

Options:
    --deception-type=<type>           'ambiguity' or 'exaggeration' [default: ambiguity]
    --action=<action>                 'compare-decoy-switch-time' or 'compare-visibility-radius' or 'animate' [default: compare-decoy-switch-time]
    --seed=<seed>                     Random seed for graph generation [default: 51]
"""

from io import BytesIO

import adjustText
import cv2
import docopt
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.spatial as spatial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.PoissonDisk.html
import scipy.stats.qmc as qmc
import torch
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
import tqdm
from scipy import interpolate

import checkpoints


def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def get_subgraph_nodes(node_locations, current_loc, visibility_radius):
    return torch.tensor([
        i for i in range(len(node_locations))
        if distance(node_locations[i], current_loc) <= visibility_radius
    ])

def get_path(
    model,
    start_node,
    node_locations,
    edge_list,
    remaining_time,
    visibility_radius,
    goal,
    decoy,
    decoy2,
    decoy2_switch_time,
    block_paths_with_velocity=False
):
    graph = torch_geometric.data.Data(edge_index=torch.tensor(edge_list.T.astype(np.int64)))
    graph: torch_geometric.data.Data = torch_geometric.transforms.ToUndirected()(graph) # type: ignore
    path = [start_node]
    current_node = start_node
    current_loc = node_locations[current_node]
    min_dist = min(((node_locations-goal)**2).sum(-1))**0.5
    curr_path_length = 0

    d_goal = []
    d_decoy = []

    switched = False
    switch_location = [0, 0]

    while distance(current_loc, goal) > min_dist:
        if curr_path_length > decoy2_switch_time and not switched:
            decoy = decoy2
            switched = True
            switch_location = current_loc
        new_edge_index = torch_geometric.utils.subgraph(get_subgraph_nodes(node_locations, current_loc, visibility_radius), graph.edge_index)[0]
        subgraph = torch_geometric.data.Data(edge_index=new_edge_index)
        subgraph['evader_visited'] = torch.tensor([1 if i in path else 0 for i in range(len(node_locations))])
        subgraph['remaining_time'] = torch.tensor([remaining_time] * len(node_locations))
        subgraph['goal_distance'] = torch.tensor([distance(node_locations[i], goal) for i in range(len(node_locations))])
        subgraph['decoy_distance'] = torch.tensor([distance(node_locations[i], decoy) for i in range(len(node_locations))])
        result = model.forward(subgraph, current_node)
        if len(path) > 1 and block_paths_with_velocity:
            # choose allowable neighbor nodes based on velocity
            unit_velocity = (node_locations[path[-1]] - node_locations[path[-2]]) / distance(node_locations[path[-1]], node_locations[path[-2]])
            relative_position_vectors = (node_locations - node_locations[path[-1]])
            unit_relative_position_vectors = relative_position_vectors/np.linalg.norm(relative_position_vectors, axis=1, keepdims=True)
            coses = np.sum(unit_velocity * unit_relative_position_vectors, axis=1)
            min_cos = 0.4
            unallowable = np.where(coses < min_cos)[0]
            for n in unallowable:
                # if all are -np.inf, then we will still be able to pick one
                result.logits[result.neighbor_nodes==n] = -np.inf
        next_node = result.neighbor_nodes[torch.argmax(result.logits)]
        path.append(next_node)
        current_node = next_node
        current_loc = node_locations[current_node]
        remaining_time -= distance(node_locations[path[-1]], node_locations[path[-2]])
        curr_path_length += distance(node_locations[path[-1]], node_locations[path[-2]])
        d_goal.append(distance(current_loc, goal))
        d_decoy.append(distance(current_loc, decoy))

    if remaining_time <= 0:
        print("overflow", -remaining_time)

    return path, switch_location

def make_base_plot(trees, start, goal, decoy, decoy2, vor, plot_voronoi=True):
    if plot_voronoi:
        spatial.voronoi_plot_2d(vor)
    else:
        plt.scatter(trees[:,0], trees[:,1], c='grey', s=4) # , label='Trees')
    plt.scatter([start[0]], [start[1]], c='r', label='Start')
    plt.scatter([goal[0]], [goal[1]], c='g', label='Goal')
    plt.scatter([decoy[0]], [decoy[1]], c='b', label='Decoy')
    plt.scatter([decoy2[0]], [decoy2[1]], c='orange', label='Decoy 2')

def spline(x,y,steps_per_edge=20):
    tck, u     = interpolate.splprep( [x,y] ,s = 0 )
    xnew,ynew = interpolate.splev( np.linspace( 0, 1, steps_per_edge * (len(x) - 1) ), tck,der = 0)
    return xnew, ynew

def label(x,y,t,bg='grey',fg='white',fontsize=12, labelalpha=0.5):
    t = plt.text(x, y, t, fontsize=fontsize, color=fg)
    t.set_bbox(dict(facecolor=bg, alpha=labelalpha, edgecolor=bg))
    return t

def no_margins():
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0.02, 0.02)

# Rendering a nice animation
def render_animation(
    model,
    start_node,
    goal_node,
    decoy_node,
    decoy_node_2,
    decoy_node_switch_time,
    node_locations,
    edge_list,
    deception_type,
    extra_time,
    visibility_radius,
    trees,
    vor,
):
    size = (1200, 1200)
    writer = cv2.VideoWriter(f'animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    graph = torch_geometric.data.Data(edge_index=torch.tensor(edge_list.T.astype(np.int64)))
    graph: torch_geometric.data.Data = torch_geometric.transforms.ToUndirected()(graph) # type: ignore

    path, switch_location = get_path(
        model,
        start_node,
        node_locations,
        edge_list,
        extra_time,
        visibility_radius,
        goal_node,
        decoy_node,
        decoy_node_2,
        decoy2_switch_time=decoy_node_switch_time
    )

    start = node_locations[start_node]

    for i in tqdm.tqdm(range(2, len(path) + 1), desc='Rendering animation'):
        make_base_plot(trees, start, goal_node, decoy_node, decoy_node_2, vor, plot_voronoi=False)

        current_node = path[i - 1]
        
        # plot path
        x = [node_locations[i][0] for i in path[:i]]
        y = [node_locations[i][1] for i in path[:i]]
        
        plt.plot(x, y)
        subgraph_nodes = get_subgraph_nodes(node_locations, node_locations[current_node], visibility_radius)
        new_edge_index = torch_geometric.utils.subgraph(subgraph_nodes, graph.edge_index)[0]
        for u,v in new_edge_index.T:
            plt.plot([node_locations[u][0], node_locations[v][0]], [node_locations[u][1], node_locations[v][1]], c='pink', linewidth=1)
        plt.plot([node_locations[i][0] for i in subgraph_nodes], [node_locations[i][1] for i in subgraph_nodes], 'o', c='purple', markersize=4)
        plt.plot([node_locations[current_node][0]], [node_locations[current_node][1]], 'o', c='orange', markersize=8)

        label(start[0]-2.5, start[1], 'Start',fontsize=10)
        label(goal_node[0]-2.5, goal_node[1], 'Goal',fontsize=10)
        label(decoy_node[0]+1, decoy_node[1], 'Decoy',fontsize=10)
        plt.axis('off')
        no_margins()
        # savefig to buffer
        io = BytesIO()
        plt.savefig(io, format='png', pad_inches=0)
        io.seek(0)
        # read from buffer to image
        img = PIL.Image.open(io).resize(size)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # write to video
        writer.write(img)
        plt.close()
    writer.release()

def main():
    args = docopt.docopt(__doc__)

    mode = args['--action']
    deception_type = args['--deception-type']
    seed = int(args['--seed'])

    engine = qmc.PoissonDisk(2, radius=0.05, seed=0)
    trees = engine.random(200) * 20
    voronoi = spatial.Voronoi(trees)

    np.random.seed(seed)
    start_node, goal_node, decoy_node, decoy_node2 = np.random.choice(len(voronoi.vertices), 4, replace=False)
    start = voronoi.vertices[start_node]
    goal = voronoi.vertices[goal_node]
    decoy = voronoi.vertices[decoy_node]
    decoy2 = voronoi.vertices[decoy_node2]
    in_graph = lambda pt: np.all(np.logical_and(pt >= 0, pt <= 20))
    min_dist = 12
    # ensure nodes are far away enough from each other to be interesting
    while distance(start,goal) < min_dist or distance(start,decoy) < min_dist or distance(goal,decoy) < min_dist or distance(start,decoy2) < min_dist or not (in_graph(start) and in_graph(goal) and in_graph(decoy) and in_graph(decoy2)):
        start_node, goal_node, decoy_node, decoy_node2 = np.random.choice(len(voronoi.vertices), 4, replace=False)
        start = voronoi.vertices[start_node]
        goal = voronoi.vertices[goal_node]
        decoy = voronoi.vertices[decoy_node]
        decoy2 = voronoi.vertices[decoy_node2]

    decoy_node, goal_node, decoy_node2 = decoy_node, decoy_node2, goal_node
    goal = voronoi.vertices[goal_node]
    decoy = voronoi.vertices[decoy_node]
    decoy2 = voronoi.vertices[decoy_node2]
    decoy2_switch_time = 4.3

    node_locations = voronoi.vertices
    edge_list = np.array(voronoi.ridge_vertices)
    edge_list = edge_list[(edge_list[:,0] >= 0) & (edge_list[:,1] >= 0)]

    # keys:
    #     'evader_visited',
    #     'goal_distance',
    #     'decoy_distance',
    #     'remaining_time',

    if deception_type == 'exaggeration':
        print("Running in mode: Exaggeration")
        model = checkpoints.load_checkpoint(checkpoints.models[1])
    else:
        print("Running in mode: Ambiguity")
        model = checkpoints.load_checkpoint(checkpoints.models[0])

    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.dpi'] = 200
    plt.title(f"{deception_type.capitalize()}: Permitted deception")
    min_ = 10
    max_ = 50
    samples = 6
    texts = []
    colors = ['red', 'green', 'blue', 'orange', 'pink', 'gold'] * 3
    labelcolors = ['white', 'white', 'white', 'black', 'black', 'black'] * 3
    fontsize = 12
    labelalpha = 0.75
    # plt.show()

    if mode == 'compare-decoy-switch-time':
        make_base_plot(trees, start, goal, decoy, decoy2, voronoi, plot_voronoi=False)

        for i, decoy2_switch_time in enumerate([1, 4, 9, 15, 21]):
            remaining_time = 30
            path, switch_location = get_path(
                model,
                start_node,
                node_locations,
                edge_list,
                remaining_time,
                2,
                goal,
                decoy,
                decoy2,
                decoy2_switch_time=decoy2_switch_time
            )

            # plot path
            x = [node_locations[j][0] for j in path]
            y = [node_locations[i][1] for i in path]
            # add label next to switch location
            t = plt.text(switch_location[0], switch_location[1], f'{round(decoy2_switch_time)}', fontsize=fontsize, color=labelcolors[i])
            t.set_bbox(dict(facecolor=colors[i], alpha=labelalpha, edgecolor=colors[i]))
            texts.append(t)
            
            plt.plot(*spline(x,y),c=colors[i], zorder=1)
            plt.scatter([switch_location[0]], [switch_location[1]], c='black', marker='h', s=20, zorder=2)

        t1 = label(start[0]-2, start[1], 'Start', fontsize=10)
        t2 = label(goal[0] - 0.5, goal[1] - 1.5, 'Goal', fontsize=10)
        t3 = label(decoy[0]+2, decoy[1]+0.5, 'Initial decoy', fontsize=10)
        t4 = label(decoy2[0]-2, decoy2[1]+0.5, 'New decoy', fontsize=10)
        plt.axis('off')
        # plt.legend(loc='upper left')
        adjustText.adjust_text([t1, t2, t3, t4])
        no_margins()
        # plt.savefig(f'figures/forest_{deception_type}_changing_extra_distance.png', pad_inches=0)
        plt.show()

    elif mode == 'compare-visibility-radius':
        plt.title(f"{deception_type.capitalize()}: Visibility radius")
        make_base_plot(trees, start, goal, decoy, decoy2, voronoi, plot_voronoi=False)

        min_ = 1.5
        max_ = 4.5
        samples = 4
        for i, visibility_radius in enumerate(tqdm.tqdm(np.exp(np.linspace(np.log(min_), np.log(max_), samples)))):
            path, switch_time = get_path(
                model,
                start_node,
                node_locations,
                edge_list,
                20,
                visibility_radius,
                goal,
                decoy,
                decoy2,
                decoy2_switch_time=decoy2_switch_time
            )

            # plot path
            x = [node_locations[i][0] for i in path]
            y = [node_locations[i][1] for i in path]
            # add label to middle of path
            t = plt.text(x[len(x)//2], y[len(y)//2], f'{visibility_radius:.2f}', fontsize=fontsize, color=labelcolors[i])
            texts.append(t)
            t.set_bbox({
                "facecolor": colors[i],
                "alpha": labelalpha,
                "edgecolor": colors[i]
            })
            
            plt.plot(*spline(x,y),c=colors[i])

        label(start[0]-2.5, start[1], 'Start',fontsize=10)
        label(goal[0]-2.5, goal[1], 'Goal',fontsize=10)
        label(decoy[0]+1, decoy[1], 'Decoy',fontsize=10)
        plt.axis('off')
        adjustText.adjust_text(texts)
        no_margins()
        # plt.savefig(f'figures/forest_{deception_type}_changing_visibility.png', pad_inches=0)
        plt.show()

    elif mode == 'animate':
        extra_time = 25
        visibility_radius = 4
        render_animation(
            model,
            start_node,
            goal,
            decoy,
            decoy2,
            decoy2_switch_time,
            node_locations,
            edge_list,
            deception_type,
            extra_time,
            visibility_radius,
            trees,
            voronoi
        )

    else:
        print("Invalid action selected.")

if __name__ == '__main__':
    main()
