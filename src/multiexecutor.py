"""
Usage: multiexecutor.py [options]

Options:
    --layers=<num>              Number of layers to use for the model.
    --deception_type=<type>     Type of deception to use. One of 'exaggeration' or 'ambiguity'.

"""

from subprocess import Popen
import sys
import docopt

args = docopt.docopt(__doc__)

def format_command(args_dict):
    command = 'python train_for_deceptiveness.py'
    for key, value in args_dict.items():
        command += f' --{key}={value}'
    return command

def v28_ablation():
    commands = []

    # Testing different numbers of layers
    layers = int(args['--layers'])
    deception_type = args['--deception_type']
    model_type = 'sage'
    for seed in range(5):
        # No prioritized training: num_prioritized_training_scenarios = -1
        commands.append(format_command({
            'feed_forward_layers': layers,
            'graph_depth': layers,
            'deception_type': deception_type,
            'model_type': model_type,
            'distance_metric': "shortest_path",
            'num_prioritized_training_scenarios': -1,
            'num_batches': 96 * 4,
            'seed': seed,
            'render_every': 48,
            'custom_key': f'v28_ablation/no_prioritized_training_{deception_type}_{layers}',
        }))

        # No PPO: copy_every = -1
        commands.append(format_command({
            'feed_forward_layers': layers,
            'graph_depth': layers,
            'deception_type': deception_type,
            'model_type': model_type,
            'distance_metric': "shortest_path",
            'num_prioritized_training_scenarios': 16,
            'num_batches': 96 * 4,
            'seed': seed,
            'render_every': 48,
            'copy_every': -1,
            'custom_key': f'v28_ablation/no_ppo_{deception_type}_{layers}',
        }))

    processes = [Popen(cmd.split(), stderr=sys.stderr) for cmd in commands]

    for process in processes:
        process.wait()

def v29_alt_graph_models():
    commands = []

    # Testing different numbers of layers
    layers = int(args['--layers'])
    deception_type = args['--deception_type']
    for model_type in ['gnn.gat', 'gnn.gcn', 'gnn.gin', 'gnn.sage']:
        for seed in range(5):
            # No prioritized training: num_prioritized_training_scenarios = -1
            commands.append(format_command({
                'feed_forward_layers': layers,
                'graph_depth': layers,
                'deception_type': deception_type,
                'model_type': model_type,
                'distance_metric': "shortest_path",
                'num_prioritized_training_scenarios': -1,
                'num_batches': 96 * (4 + 1),
                'continue_from_step': (96 * 4) * 256,
                'seed': seed,
                'render_every': 48,
                'custom_key': f'v29_alt_graph_models_no_prioritized_training/{model_type}_{deception_type}_{layers}',
            }))

    processes = [Popen(cmd.split(), stderr=sys.stderr) for cmd in commands]

    for process in processes:
        process.wait()

def v30_nullify_if_unsuccessful():
    commands = []

    # Testing different numbers of layers
    layers = int(args['--layers'])
    deception_type = args['--deception_type']
    for model_type in ['gnn.gat', 'gnn.gcn', 'gnn.gin', 'gnn.sage']:
        for seed in range(1):
            # No prioritized training: num_prioritized_training_scenarios = -1
            commands.append(format_command({
                'feed_forward_layers': layers,
                'graph_depth': layers,
                'deception_type': deception_type,
                'model_type': model_type,
                'distance_metric': "shortest_path",
                'num_prioritized_training_scenarios': -1,
                'num_batches': 96 * 6,
                'continue_from_step': (96 * 4) * 256,
                'seed': seed,
                'render_every': 48,
                'nullify_if_unsuccessful': 1,
                'custom_key': f'v30_nullify_if_unsuccessful/{model_type}_{deception_type}_{layers}_prioritized',
            }))
            # Yes prioritized training: num_prioritized_training_scenarios = 256
            commands.append(format_command({
                'feed_forward_layers': layers,
                'graph_depth': layers,
                'deception_type': deception_type,
                'model_type': model_type,
                'distance_metric': "shortest_path",
                'num_prioritized_training_scenarios': 256,
                'num_batches': 96 * 6,
                'continue_from_step': (96 * 4) * 256,
                'seed': seed,
                'render_every': 48,
                'nullify_if_unsuccessful': 1,
                'custom_key': f'v30_nullify_if_unsuccessful/{model_type}_{deception_type}_{layers}_unprioritized',
            }))

    processes = [Popen(cmd.split(), stderr=sys.stderr) for cmd in commands]

    for process in processes:
        process.wait()

def v31_use_constant_deceptiveness():
    commands = []

    # Testing different numbers of layers
    layers = int(args['--layers'])
    deception_type = args['--deception_type']
    for model_type in ['gnn.gat', 'gnn.gcn', 'gnn.gin', 'gnn.sage']:
        for seed in range(1):
            # No prioritized training: num_prioritized_training_scenarios = -1
            commands.append(format_command({
                'feed_forward_layers': layers,
                'graph_depth': layers,
                'deception_type': deception_type,
                'model_type': model_type,
                'distance_metric': "shortest_path",
                'num_prioritized_training_scenarios': -1,
                'num_batches': 96 * 4,
                # 'continue_from_step': (96 * 4) * 256,
                'seed': seed,
                'render_every': 48,
                'nullify_if_unsuccessful': 1,
                'use_constant_deceptiveness': 1,
                'custom_key': f'v31_use_constant_deceptiveness/{model_type}_{deception_type}_{layers}_prioritized',
            }))
            # Yes prioritized training: num_prioritized_training_scenarios = 256
            commands.append(format_command({
                'feed_forward_layers': layers,
                'graph_depth': layers,
                'deception_type': deception_type,
                'model_type': model_type,
                'distance_metric': "shortest_path",
                'num_prioritized_training_scenarios': 256,
                'num_batches': 96 * 4,
                # 'continue_from_step': (96 * 4) * 256,
                'seed': seed,
                'render_every': 48,
                'nullify_if_unsuccessful': 1,
                'use_constant_deceptiveness': 1,
                'custom_key': f'v31_use_constant_deceptiveness/{model_type}_{deception_type}_{layers}_unprioritized',
            }))

    processes = [Popen(cmd.split(), stderr=sys.stderr) for cmd in commands]

    for process in processes:
        process.wait()

v31_use_constant_deceptiveness()
