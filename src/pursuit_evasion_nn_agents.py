import random
from dataclasses import dataclass

import networkx
import torch
import torch_geometric
import torch_geometric.nn
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.transforms.add_positional_encoding import \
    AddLaplacianEigenvectorPE


@dataclass
class GraphModelOutput:
    logits: torch.Tensor
    value: torch.Tensor
    neighbor_nodes: list

class FixedPositionalEncoding(torch.nn.Module):
    def __init__(self, num_locations: int, dim: int):
        super().__init__()

        self.position_embedding = torch.nn.Embedding(num_locations, dim)

    def forward(self, graph: Data):
        graph.positional_encoding = pos_enc = self.position_embedding.weight.clone()
        graph['x'] = pos_enc
        return graph

class LaplacianEigenvectorPositionalEncoding(torch.nn.Module):
    def __init__(self, k: int, dim: int, is_undirected=True):
        super().__init__()

        self.pos_enc = AddLaplacianEigenvectorPE(k, attr_name='positional_encoding', is_undirected=is_undirected)
        self.linear = torch.nn.Linear(k, dim)

    def forward(self, graph: Data):
        self.pos_enc(graph)
        graph['x'] = self.linear(graph.positional_encoding)
        return graph

AddLaplacianEigenvectorPE(k=8, attr_name='positional_encoding', is_undirected=True)

class PursuitGraphModel(torch.nn.Module):
    def __init__(self, dim=64, layers=4, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True, dropout=0.1, modeling_approach='sage'):
        super().__init__()

        if modeling_approach == 'gat':
            GraphModelModule = torch_geometric.nn.GAT
        elif modeling_approach == 'sage':
            GraphModelModule = torch_geometric.nn.GraphSAGE
        elif modeling_approach == 'gcn':
            GraphModelModule = torch_geometric.nn.GCN
        elif modeling_approach == 'gin':
            GraphModelModule = torch_geometric.nn.GIN
        else:
            raise ValueError(f"Unknown modeling approach: {modeling_approach}")

        # Input channels are {pursuer?, evader?}
        # Output channels are {policy, value}
        self.graph_sage = GraphModelModule(
            in_channels=len(keys),
            hidden_channels=dim,
            num_layers=layers,
            out_channels=dim,
            dropout=dropout,
        )
        self.num_layers = layers
        self.logit_projection = torch.nn.Linear(dim * 2, 1)
        self.value_projection = torch.nn.Linear(dim, 1)
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node):
        from graph_feature_engineering import \
            networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping

        subgraph = networkx.ego_graph(nx_graph, self_node, radius=self.num_layers + 1)
        torchgraph, mapping, inverse_mapping = networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping(subgraph)

        output = self.forward(torchgraph, mapping[self_node])

        # Translate output graph labels
        return GraphModelOutput(
            logits=output.logits,
            value=output.value,
            neighbor_nodes=[inverse_mapping[i] for i in output.neighbor_nodes]
        )

    def forward(self, graph: Data, self_node: int):
        assert type(graph) == Data, "Must pass in a torch-geometric graph."

        # Construct `x`
        x = torch.stack([
            graph[key] for key in self.keys
        ], dim=-1).to(torch.float)

        embeddings = self.graph_sage.forward(x, graph.edge_index)
        self_embedding = embeddings[self_node]
        neighbor_nodes = [node.item() for node in graph.edge_index[1, graph.edge_index[0, :] == self_node]]

        # If we want to augment, augment the edge index through random edge dropping.
        if self.augment and self.training:
            mask = torch.rand(graph.edge_index.shape[1]) > 0.1
            graph['edge_index'] = graph.edge_index[:, mask]

        # Format: [dim + dim], [current node + target node]
        edge_embeddings = torch.stack([
            torch.cat((self_embedding, embeddings[neighbor_node]))
            for neighbor_node in neighbor_nodes
        ])

        logits = self.logit_projection(edge_embeddings)[:, 0]
        value = self.value_projection(self_embedding)[0]

        return GraphModelOutput(logits, value, neighbor_nodes)

def iterate_all_walks(graph: networkx.Graph, start_node, target_edge_count, visited):
    if target_edge_count == 0:
        yield [start_node]
        return
    
    found_walk = False
    visited.add(start_node)
    for neighbor in graph.neighbors(start_node):
        if neighbor in visited:
            continue
        
        for walk in iterate_all_walks(graph, neighbor, target_edge_count - 1, visited):
            yield [start_node] + walk
            found_walk = True
    visited.remove(start_node)

    # If we have no paths with the target length, just return the path so far with padding.
    if not found_walk:
        yield [start_node] + [None] * target_edge_count

def create_general_mlp(layer_dims, dropout):
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))
        layers.append(torch.nn.ReLU())
        if dropout != 0:
            layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)

class PursuitTreeSearchModel(torch.nn.Module):
    def __init__(self, dim=64, graph_depth=2, feed_forward_layers=4, dropout=0.1, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True):
        super().__init__()


        self.path_projection = torch.nn.Linear(
            in_features=(len(keys) + 1) * graph_depth,
            out_features=dim,
        )
        path_feature_dim = (len(keys) + 1) * (graph_depth)
        path_inner_dim = dim * graph_depth
        self.edge_mlp = create_general_mlp([path_feature_dim] + [path_inner_dim] * feed_forward_layers + [dim], dropout)
        self.dropout = dropout
        self.graph_depth = graph_depth
        self.num_layers = feed_forward_layers
        self.logit_projection = torch.nn.Linear(dim, 1)
        self.value_projection = torch.nn.Linear(dim, 1)
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node: int):
        from graph_feature_engineering import \
            networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping

        torchgraph, mapping, inverse_mapping = networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping(nx_graph)
        walks = list(iterate_all_walks(nx_graph, self_node, self.graph_depth - 1, set()))

        # Map walks
        walks = [[mapping[node] if node is not None else None for node in walk] for walk in walks]

        output = self.forward(torchgraph, walks)

        # Translate output graph labels
        return GraphModelOutput(
            logits=output.logits,
            value=output.value,
            neighbor_nodes=[inverse_mapping[i] for i in output.neighbor_nodes]
        )

    def forward(self, graph: Data, walks: list):
        assert type(graph) == Data, "Must pass in a torch-geometric graph."
        assert graph.num_nodes is not None, "Must pass in a torch-geometric graph with num_nodes."

        # Construct `x`, adding a 1 flag to indicate that this is not a NULL node.
        x = torch.stack([
            graph[key] for key in self.keys
        ] + [torch.zeros(graph.num_nodes)], dim=-1).to(torch.float)

        null_node = torch.zeros_like(x[0])

        # Get embeddings for all walks
        path_features = torch.stack([
            torch.cat([x[node] if node is not None else null_node for node in walk])
            for walk in walks
        ])

        path_embeddings = self.path_projection(path_features)

        # Calculate value and logits for each path
        values = self.value_projection(path_embeddings)
        logits = self.logit_projection(path_embeddings)

        # Calculate value using softmax over logits (detached so they don't backprop)
        value = (values * torch.softmax(logits, dim=0).detach()).sum()
        
        # Calculate logits, bucketed by the second node
        neighbor_node_logit_dict = {walk[1]: [] for walk in walks}
        for walk, logit in zip(walks, logits):
            neighbor_node_logit_dict[int(walk[1])].append(logit)
        
        # Calculate logits for each neighbor node
        neighbor_node_logit_dict = {node: torch.stack(logit_list).mean() for node, logit_list in neighbor_node_logit_dict.items()}
        neighbor_nodes, logits = zip(*neighbor_node_logit_dict.items())
        neighbor_nodes = list(neighbor_nodes)
        logits = torch.stack(logits)
        
        return GraphModelOutput(logits, value, neighbor_nodes)

def convert_local_subgraph_to_torch_tensor(nx_graph: networkx.Graph, self_node, size, keys):
    central_node = self_node
    grid = torch.zeros((size * 2 + 1, size * 2 + 1, len(keys) + 1))
    for node in nx_graph.nodes:
        node_y, node_x = node
        grid_y = node_y - central_node[0] + size
        grid_x = node_x - central_node[1] + size
        if 0 <= grid_y < size and 0 <= grid_x < size:
            grid[grid_y, grid_x, :-1] = torch.stack([nx_graph[node].get(key, torch.tensor(0)) for key in keys])
            # For obstacles, this value is set to 0.
            grid[grid_y, grid_x, -1] = 1
    return grid

def augment_grid(grid):
    # Returns the augmented grid: A random flip and a random 90º rotation, along with how to undo the augmentation.
    flip_dim = random.choice([-1, 0, 1])
    if flip_dim > -1:
        grid = torch.flip(grid, dims=(flip_dim,))
    rotate_k = random.randint(0, 3)
    grid = torch.rot90(grid, k=rotate_k, dims=(0, 1))

    return (grid, rotate_k, flip_dim)

# An *actual* MLP model -- assuming a grid world
class PursuitGridMLPModel(torch.nn.Module):
    def __init__(self, dim=64, graph_depth=2, feed_forward_layers=4, dropout=0.1, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True):
        super().__init__()

        self.input_grid_size = (graph_depth * 2 + 1, graph_depth * 2 + 1)
        self.input_node_count = self.input_grid_size[0] * self.input_grid_size[1]
        self.feature_count_per_node = len(keys) + 1
        self.input_feature_count = self.feature_count_per_node * self.input_node_count
        self.mlp = create_general_mlp([self.input_feature_count] + [dim] * feed_forward_layers, dropout)
        self.dropout = dropout
        self.graph_depth = graph_depth
        self.num_layers = feed_forward_layers
        # Up, down, left, right
        self.logit_projection = torch.nn.Linear(dim, 4)
        self.value_projection = torch.nn.Linear(dim, 1)
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node):
        grid = convert_local_subgraph_to_torch_tensor(nx_graph, self_node, self.graph_depth, self.keys)
        central_node = self_node

        if self.augment:
            # Augment the grid
            grid_aug, rotate_k, flip_dim = augment_grid(grid)
            output = self.forward(grid_aug)
            # output_normal = self.forward(grid)
            up_logit, down_logit, left_logit, right_logit = output.logits
            if rotate_k == 1:
                # rotate 1 clockwise
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
            elif rotate_k == 2:
                # rotate 2 clockwise
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, right_logit, left_logit
            elif rotate_k == 3:
                # rotate 3 clockwise
                up_logit, down_logit, left_logit, right_logit = right_logit, left_logit, down_logit, up_logit
            if flip_dim == 0:
                # flip vertically
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, left_logit, right_logit
            elif flip_dim == 1:
                # flip horizontally
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
        
            logits = torch.tensor([up_logit, down_logit, left_logit, right_logit])
        else:
            output = self.forward(grid)
            logits = output.logits

        up_node = (central_node[0] - 1, central_node[1])
        down_node = (central_node[0] + 1, central_node[1])
        left_node = (central_node[0], central_node[1] - 1)
        right_node = (central_node[0], central_node[1] + 1)
        available_actions = [up_node, down_node, left_node, right_node]
        reachable_mask = [nx_graph.has_node(node) for node in available_actions]

        # Translate output graph labels
        return GraphModelOutput(
            logits=logits[reachable_mask],
            value=output.value[0],
            neighbor_nodes=[available_actions[i] for i in range(4) if reachable_mask[i]]
        )

    def forward(self, grid: torch.Tensor):
        assert type(grid) == torch.Tensor, "Must pass in a torch tensor."

        x = grid.flatten()
        x = self.mlp(x)

        # Calculate value and logits
        value = self.value_projection(x)
        logits = self.logit_projection(x)
        
        return GraphModelOutput(logits, value, [])

class PursuitGridMLPModelSeparateNetworks(torch.nn.Module):
    def __init__(self, dim=64, graph_depth=2, feed_forward_layers=4, dropout=0.1, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True):
        super().__init__()

        self.input_grid_size = (graph_depth * 2 + 1, graph_depth * 2 + 1)
        self.input_node_count = self.input_grid_size[0] * self.input_grid_size[1]
        self.feature_count_per_node = len(keys) + 1
        self.input_feature_count = self.feature_count_per_node * self.input_node_count
        self.value_network = create_general_mlp([self.input_feature_count] + [dim] * feed_forward_layers + [1], dropout)
        self.policy_network = create_general_mlp([self.input_feature_count] + [dim] * feed_forward_layers + [4], dropout)
        self.dropout = dropout
        self.graph_depth = graph_depth
        self.num_layers = feed_forward_layers
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node):
        grid = convert_local_subgraph_to_torch_tensor(nx_graph, self_node, self.graph_depth, self.keys)
        central_node = self_node

        if self.augment:
            # Augment the grid
            grid_aug, rotate_k, flip_dim = augment_grid(grid)
            output = self.forward(grid_aug)
            # output_normal = self.forward(grid)
            up_logit, down_logit, left_logit, right_logit = output.logits
            if rotate_k == 1:
                # rotate 1 clockwise
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
            elif rotate_k == 2:
                # rotate 2 clockwise
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, right_logit, left_logit
            elif rotate_k == 3:
                # rotate 3 clockwise
                up_logit, down_logit, left_logit, right_logit = right_logit, left_logit, down_logit, up_logit
            if flip_dim == 0:
                # flip vertically
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, left_logit, right_logit
            elif flip_dim == 1:
                # flip horizontally
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
        
            logits = torch.tensor([up_logit, down_logit, left_logit, right_logit])
        else:
            output = self.forward(grid)
            logits = output.logits

        up_node = (central_node[0] - 1, central_node[1])
        down_node = (central_node[0] + 1, central_node[1])
        left_node = (central_node[0], central_node[1] - 1)
        right_node = (central_node[0], central_node[1] + 1)
        available_actions = [up_node, down_node, left_node, right_node]
        reachable_mask = [nx_graph.has_node(node) for node in available_actions]

        # Translate output graph labels
        return GraphModelOutput(
            logits=logits[reachable_mask],
            value=output.value,
            neighbor_nodes=[available_actions[i] for i in range(4) if reachable_mask[i]]
        )

    def forward(self, grid: torch.Tensor):
        assert type(grid) == torch.Tensor, "Must pass in a torch tensor."

        x = grid.flatten()
        value = self.value_network(x).squeeze(-1)
        logits = self.policy_network(x)
        
        return GraphModelOutput(logits, value, [])

# Trying "residual" approach -- projecting to a smaller layer and then reprojecting back
class PursuitGridMLPModelResidual(torch.nn.Module):
    def __init__(self, dim=64, graph_depth=2, feed_forward_layers=4, dropout=0.1, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True):
        super().__init__()

        self.input_grid_size = (graph_depth * 2 + 1, graph_depth * 2 + 1)
        self.input_node_count = self.input_grid_size[0] * self.input_grid_size[1]
        self.feature_count_per_node = len(keys) + 1
        self.input_feature_count = self.feature_count_per_node * self.input_node_count
        self.value_network = create_general_mlp([self.input_feature_count] + [dim] * feed_forward_layers + [1], dropout)
        self.policy_network = create_general_mlp([self.input_feature_count] + [dim] * feed_forward_layers + [4], dropout)
        self.dropout = dropout
        self.graph_depth = graph_depth
        self.num_layers = feed_forward_layers
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node):
        grid = convert_local_subgraph_to_torch_tensor(nx_graph, self_node, self.graph_depth, self.keys)
        central_node = self_node

        if self.augment:
            # Augment the grid
            grid_aug, rotate_k, flip_dim = augment_grid(grid)
            output = self.forward(grid_aug)
            # output_normal = self.forward(grid)
            up_logit, down_logit, left_logit, right_logit = output.logits
            if rotate_k == 1:
                # rotate 1 clockwise
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
            elif rotate_k == 2:
                # rotate 2 clockwise
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, right_logit, left_logit
            elif rotate_k == 3:
                # rotate 3 clockwise
                up_logit, down_logit, left_logit, right_logit = right_logit, left_logit, down_logit, up_logit
            if flip_dim == 0:
                # flip vertically
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, left_logit, right_logit
            elif flip_dim == 1:
                # flip horizontally
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
        
            logits = torch.tensor([up_logit, down_logit, left_logit, right_logit])
        else:
            output = self.forward(grid)
            logits = output.logits

        up_node = (central_node[0] - 1, central_node[1])
        down_node = (central_node[0] + 1, central_node[1])
        left_node = (central_node[0], central_node[1] - 1)
        right_node = (central_node[0], central_node[1] + 1)
        available_actions = [up_node, down_node, left_node, right_node]
        reachable_mask = [nx_graph.has_node(node) for node in available_actions]

        # Translate output graph labels
        return GraphModelOutput(
            logits=logits[reachable_mask],
            value=output.value,
            neighbor_nodes=[available_actions[i] for i in range(4) if reachable_mask[i]]
        )

    def forward(self, grid: torch.Tensor):
        assert type(grid) == torch.Tensor, "Must pass in a torch tensor."

        x = grid.flatten()
        value = self.value_network(x).squeeze(-1)
        logits = self.policy_network(x)
        
        return GraphModelOutput(logits, value, [])

class PaddedResidualBlock(torch.nn.Module):
    def __init__(self, input_channels, intermediate_channels):
        super().__init__()

        self.conv1x1 = torch.nn.Conv2d(input_channels, intermediate_channels, kernel_size=1)
        self.conv3x3 = torch.nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1)
        self.conv1x1_2 = torch.nn.Conv2d(intermediate_channels, input_channels, kernel_size=1)

    def forward(self, x):
        x_orig = x
        x = self.conv1x1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3x3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv1x1_2(x)
        x = x + x_orig
        return torch.nn.functional.relu(x)

class PursuitResNetModel(torch.nn.Module):
    def __init__(self, channels, blocks, graph_depth, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness'],
                 augment=True):
        super().__init__()

        self.input_grid_size = (graph_depth * 2 + 1, graph_depth * 2 + 1)
        self.feature_count_per_node = len(keys) + 1
        self.project_to_intermediate = torch.nn.Conv2d(self.feature_count_per_node, channels, kernel_size=1)
        self.blocks = torch.nn.ModuleList([
            PaddedResidualBlock(channels, channels) for _ in range(blocks)
        ])
        self.graph_depth = graph_depth
        # Up, down, left, right
        self.logit_projection = torch.nn.Linear(channels, 1)
        self.value_projection = torch.nn.Linear(channels, 1)
        self.augment = augment
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node):
        grid = convert_local_subgraph_to_torch_tensor(nx_graph, self_node, self.graph_depth, self.keys)
        central_node = self_node

        # batchify
        grid = grid.unsqueeze(0)

        if self.augment:
            # Augment the grid
            grid_aug, rotate_k, flip_dim = augment_grid(grid)
            output = self.forward(grid_aug)
            # output_normal = self.forward(grid)
            up_logit, down_logit, left_logit, right_logit = output.logits
            if rotate_k == 1:
                # rotate 1 clockwise
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
            elif rotate_k == 2:
                # rotate 2 clockwise
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, right_logit, left_logit
            elif rotate_k == 3:
                # rotate 3 clockwise
                up_logit, down_logit, left_logit, right_logit = right_logit, left_logit, down_logit, up_logit
            if flip_dim == 0:
                # flip vertically
                up_logit, down_logit, left_logit, right_logit = down_logit, up_logit, left_logit, right_logit
            elif flip_dim == 1:
                # flip horizontally
                up_logit, down_logit, left_logit, right_logit = left_logit, right_logit, up_logit, down_logit
        
            logits = torch.tensor([up_logit, down_logit, left_logit, right_logit])
        else:
            output = self.forward(grid)
            logits = output.logits

        up_node = (central_node[0] - 1, central_node[1])
        down_node = (central_node[0] + 1, central_node[1])
        left_node = (central_node[0], central_node[1] - 1)
        right_node = (central_node[0], central_node[1] + 1)
        available_actions = [up_node, down_node, left_node, right_node]
        reachable_mask = [nx_graph.has_node(node) for node in available_actions]

        # Translate output graph labels
        return GraphModelOutput(
            logits=logits[0][reachable_mask],
            value=output.value[0],
            neighbor_nodes=[available_actions[i] for i in range(4) if reachable_mask[i]]
        )

    def forward(self, grid: torch.Tensor):
        assert type(grid) == torch.Tensor, "Must pass in a torch tensor."

        x = grid.permute(0, 3, 1, 2)
        x = self.project_to_intermediate(x)
        for block in self.blocks:
            x = block(x)

        up_embed = x[:, :, self.graph_depth - 1, self.graph_depth]
        down_embed = x[:, :, self.graph_depth + 1, self.graph_depth]
        left_embed = x[:, :, self.graph_depth, self.graph_depth - 1]
        right_embed = x[:, :, self.graph_depth, self.graph_depth + 1]
        ordered_embeds = torch.stack([
            up_embed, down_embed, left_embed, right_embed
        ], dim=-2)
        center_embedding = x[:, :, self.graph_depth, self.graph_depth]

        # Calculate value and logits
        value = self.value_projection(center_embedding).squeeze(-1)
        logits = self.logit_projection(ordered_embeds).squeeze(-1)
        
        return GraphModelOutput(logits, value, [])

"""
class PursuitMonteCarlo(torch.nn.Module):
    def __init__(self, dim, sample_count, keys=['has_evader', 'has_pursuer', 'goal_distance', 'decoy_distance', 'remaining_time', 'deceptiveness']):
        super().__init__()

        self.node_embedder = torch.nn.Linear(len(keys), dim)
        # Edge predictors
        self.logit_projection = torch.nn.Linear(dim * 2, 1)
        self.value_projection = torch.nn.Linear(dim * 2, 1)
        self.sample_count = sample_count
        self.keys = keys

    def networkx_forward(self, nx_graph: networkx.Graph, self_node: int):
        from graph_feature_engineering import networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping

        torchgraph, mapping, inverse_mapping = networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping(nx_graph)
        output = self.forward(torchgraph, self_node)

        # Translate output graph labels
        return GraphModelOutput(
            logits=output.logits,
            value=output.value,
            neighbor_nodes=[inverse_mapping[i] for i in output.neighbor_nodes]
        )

    def forward(self, graph: Data, self_node: int):
        assert type(graph) == Data, "Must pass in a torch-geometric graph."

        # Construct `x`, adding a 1 flag to indicate that this is not a NULL node.
        x = torch.stack([
            graph[key] for key in self.keys
        ], dim=-1).to(torch.float)

        # Get embeddings for all walks
        edge_features = torch.stack([
            torch.cat([x[self_node], x[neighbor]])
            for neighbor in graph.edge_index[1, graph.edge_index[0, :] == self_node]
        ])

        logits = self.logit_projection(edge_features)
        values = self.value_projection(edge_features)
        
        # Calculate value using softmax over logits (detached so they don't backprop)
        value = (values * torch.softmax(logits, dim=0).detach()).sum()
        
        # Calculate logits, bucketed by the second node
        neighbor_node_logit_dict = {walk[1]: [] for walk in walks}
        for walk, logit in zip(walks, logits):
            neighbor_node_logit_dict[int(walk[1])].append(logit)
        
        # Calculate logits for each neighbor node
        neighbor_node_logit_dict = {node: torch.stack(logit_list).mean() for node, logit_list in neighbor_node_logit_dict.items()}
        neighbor_nodes, logits = zip(*neighbor_node_logit_dict.items())
        neighbor_nodes = list(neighbor_nodes)
        logits = torch.stack(logits)
        
        return GraphModelOutput(logits, value, neighbor_nodes)
"""
