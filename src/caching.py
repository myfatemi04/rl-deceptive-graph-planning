import hashlib
import os
import pickle
from typing import TypeVar

import networkx as nx
import numpy as np


# Create hash of networkx graph by hashing a consistent version of the adjacency list.
def create_adjacency_hash(graph: nx.Graph):
    adjacency_list = []
    for node in sorted(graph.nodes):
        adjacency_list.append([
            node,
            sorted(graph.neighbors(node))
        ])
    
    adjacency_string = str(adjacency_list).encode('utf-8')
    return hashlib.sha256(adjacency_string).hexdigest()

T = TypeVar('T')

def create_cache_key(item):
    value = None
    if isinstance(item, (str, int, float, bool)):
        value = item
    elif isinstance(item, nx.Graph):
        value = create_adjacency_hash(item)
    elif isinstance(item, (tuple, list)):
        value = tuple([create_cache_key(i) for i in item])
    elif isinstance(item, dict):
        value = tuple([(create_cache_key(k), create_cache_key(v)) for k, v in sorted(item.items())])
    elif isinstance(item, set):
        value = tuple([create_cache_key(i) for i in sorted(item)])
    elif isinstance(item, np.ndarray):
        shape = tuple(item.shape)
        values = tuple(item.flatten())
        value = (shape, values)
    elif 'numpy.' in str(type(item)):
        value = item
    else:
        raise ValueError(f"Cannot create cache key from item of type {type(item)}")
    
    return (str(type(item)), value)

def pickle_cached(index_name: str):
    def decorator(fn: T) -> T:
        assert callable(fn), "Must pass in a function."

        def wrapped(*args, **kwargs):
            cache_key = hashlib.sha256(str(create_cache_key((args, kwargs))).encode('utf-8')).hexdigest()

            if not os.path.exists(f'cache/{index_name}'):
                os.makedirs(f'cache/{index_name}')

            if os.path.exists(f'cache/{index_name}/{cache_key}'):
                try:
                    with open(f'cache/{index_name}/{cache_key}', 'rb') as f:
                        stored_object = pickle.load(f)
                    
                    return stored_object['value']
                except Exception as e:
                    print("Error loading cached object.")
            
            result = fn(*args, **kwargs)
            stored_object = {
                'args': args,
                'kwargs': kwargs,
                'value': result
            }
            with open(f'cache/{index_name}/{cache_key}', 'wb') as f:
                pickle.dump(stored_object, f)
            return result

        return wrapped # type: ignore
    
    return decorator
