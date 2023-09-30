import sys

import networkx as nx
import torch
import torch_geometric.data
import torch_geometric.utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_uniform_attribute(graph: nx.Graph, attribute_name: str, value):
    nx.set_node_attributes(graph, {node: {attribute_name: value} for node in graph.nodes})

def get_proportionally_noised(value, proportion):
    if proportion == 0:
        return value
    
    if type(value) == torch.Tensor:
        return value + torch.randn(value.shape).to(device) * proportion * value
    else:
        return value + torch.randn(()) * proportion * value

def add_shortest_path_distance_labels_to_networkx_graph(graph: nx.Graph, attribute_name: str, targets: list, distance_matrix, noise):
    """
    Adds attribute `attribute_name` to each node corresponding to the shortest path distance to any of the nodes in `targets`.
    """
    
    attribute_update_dict = {
        source: {
            attribute_name: get_proportionally_noised(
                value=min([distance_matrix[source][target] for target in targets]),
                proportion=noise,
            )
        }
        for source in graph.nodes
    }
    nx.set_node_attributes(graph, attribute_update_dict)

def manhattan_distance(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def add_manhattan_distance_labels_to_networkx_graph(graph: nx.Graph, attribute_name: str, targets: list):
    """
    Adds attribute `attribute_name` to each node corresponding to the lowest Manhattan distance to any of the nodes in `targets`.
    """

    attribute_update_dict = {source: {attribute_name: min(manhattan_distance(source, target) for target in targets)} for source in graph.nodes}
    nx.set_node_attributes(graph, attribute_update_dict)

def add_position_labels_to_networkx_graph(graph: nx.Graph, attribute_name: str, nodes: list):
    node_set = set(nodes)
    nx.set_node_attributes(graph, {node: {attribute_name: node in node_set} for node in graph.nodes})

def networkx_graph_to_torch_geometric_and_mapping_and_inverse_mapping(graph: nx.Graph):
    mapping = {node: i for i, node in enumerate(graph.nodes)}
    inverse_mapping = {i: node for node, i in mapping.items()}
    
    relabeled_graph = nx.relabel_nodes(graph, mapping=mapping, copy=True)

    if len(relabeled_graph.nodes) > 0:
        # Assume that we have at least one node in the graph
        available_attributes = set(relabeled_graph.nodes[0].keys())
        
        # For each available attribute, convert to a tensor if that is the best data type
        # Otherwise, use a list.
        converted_data = {}
        for attribute in available_attributes:
            result_list = []
            attribute_type = None
            for i in range(len(relabeled_graph.nodes)):
                assert attribute in relabeled_graph.nodes[i], f"Attribute `{attribute}` not found in node {i}."
                
                value = relabeled_graph.nodes[i][attribute]
                result_list.append(value)

                if attribute_type is None:
                    attribute_type = type(value)

            try:
                result_list = torch.tensor(result_list).to(device)
            except:
                print(f"Failed to convert attribute `{attribute}` to tensor. Using list instead.", file=sys.stderr)

            converted_data[attribute] = result_list
    else:
        converted_data = {}

    edge_index = torch.tensor(list(relabeled_graph.edges)).t().contiguous().to(device)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    assert type(edge_index) == torch.Tensor

    data = torch_geometric.data.Data(edge_index=edge_index, num_nodes=relabeled_graph.number_of_nodes(), **converted_data)

    return data, mapping, inverse_mapping
