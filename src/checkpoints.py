import torch

import pursuit_evasion_nn_agents

models = [
    {'visibility': 2, 'path': '../models/sage_ambiguity_2/model.pt', 'deception_type': 'ambiguity'},
    {'visibility': 4, 'path': '../models/sage_exaggeration_4/model.pt', 'deception_type': 'exaggeration'},
]

def load_checkpoint(checkpoint):
    visibility = checkpoint['visibility']
    model = pursuit_evasion_nn_agents.PursuitGraphModel(
        dim=64,
        layers=visibility,
        keys=[
            'evader_visited',
            'goal_distance',
            'decoy_distance',
            'remaining_time',
        ],
        augment=False,
    )
    model.load_state_dict(torch.load(checkpoint['path']))
    model.eval()
    
    return model
