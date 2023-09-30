from deception_types import AmbiguousDeceptionFunction, ExaggerationDeceptionFunction
from landscape import Landscape

def measure_ambiguity(landscape: Landscape, path, true, decoy):
    fn = AmbiguousDeceptionFunction(
        landscape.value_function_table,
        {true: 0.5, decoy: 0.5},
        true,
        0.95,
        landscape.graph,
    )
    total_deceptiveness = 0
    for i in range(len(path)):
        deceptiveness_this_step = fn(path[:i + 1])
        total_deceptiveness += deceptiveness_this_step
    return total_deceptiveness

def measure_exaggeration(landscape: Landscape, path, true, decoy):
    fn = ExaggerationDeceptionFunction(
        landscape.value_function_table,
        {true: 0.5, decoy: 0.5},
        true,
        0.95,
    )
    total_deceptiveness = 0
    for i in range(len(path)):
        deceptiveness_this_step = fn(path[:i + 1])
        total_deceptiveness += deceptiveness_this_step
    return total_deceptiveness
