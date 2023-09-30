# Starts from a seed file. Then creates a dependency graph and outputs files that were not used.

import os

dependency_graph = {}

def explore_dependencies(file):
    if file in dependency_graph:
        # Already been explored.
        if dependency_graph[file] is None:
            print("Circular dependency detected:", file)
        return
    
    dependency_graph[file] = None

    dependencies = []
    with open(file + '.py') as f:
        content = f.read()
    for line in content.split("\n"):
        line = line.strip()
        if line.strip().startswith('import'):
            dependencies.append(line.strip().split()[1])
        elif line.strip().startswith('from'):
            dependencies.append(line.strip().split()[1])

    local_deps = []
    
    for dependency in dependencies:
        # Check if this file exists.
        if os.path.isfile(dependency + '.py'):
            # Recur through this file.
            explore_dependencies(dependency)
            local_deps.append(dependency)

    dependency_graph[file] = local_deps

entrypoints = [
    'train_for_deceptiveness',
    'test_large_graphs',
    'multiexecutor',
    'continuous_sim',
    'inspect_reward_shaping',
    'test_goal_switch',
    'measure_total_path_deceptiveness',
    'find_deps',
]

for initial_file in entrypoints:
    explore_dependencies(initial_file)

for f in dependency_graph:
    if 'graph_examples' in dependency_graph[f]:
        print(f)

print("UNUSED FILES:")
for file in sorted(os.listdir()):
    if file.endswith(".py"):
        name = file[:-3]
        if name not in dependency_graph.keys():
            print(f"Unused file: {name}")
