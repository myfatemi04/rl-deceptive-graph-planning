from colorsys import hsv_to_rgb

import cv2
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import tqdm

HSV = cm.get_cmap("hsv")
inferno = cm.get_cmap("inferno")


def normalize_pos_dict(pos_dict, output_size=600):
    min_x = min(pos_dict.values(), key=lambda x: x[0])[0]
    max_x = max(pos_dict.values(), key=lambda x: x[0])[0]
    min_y = min(pos_dict.values(), key=lambda x: x[1])[1]
    max_y = max(pos_dict.values(), key=lambda x: x[1])[1]
    updated_pos = {node: (pos_dict[node][0] - min_x, pos_dict[node][1] - min_y) for node in pos_dict}
    updated_pos = {node: (updated_pos[node][0] / (max_x - min_x), updated_pos[node][1] / (max_y - min_y)) for node in pos_dict}
    updated_pos = {node: ((updated_pos[node][0] * 0.8 + 0.1) * output_size, (updated_pos[node][1] * 0.8 + 0.1) * output_size) for node in pos_dict}
    return updated_pos

def normalize_pos_dict_square(pos_dict, output_size=600):
    min_x = min(pos_dict.values(), key=lambda x: x[0])[0]
    max_x = max(pos_dict.values(), key=lambda x: x[0])[0]
    min_y = min(pos_dict.values(), key=lambda x: x[1])[1]
    max_y = max(pos_dict.values(), key=lambda x: x[1])[1]
    size = max(max_x - min_x, max_y - min_y)
    updated_pos = {node: (pos_dict[node][0] - min_x, pos_dict[node][1] - min_y) for node in pos_dict}
    updated_pos = {node: (updated_pos[node][0] / size, updated_pos[node][1] / size) for node in pos_dict}
    updated_pos = {node: ((updated_pos[node][0] * 0.8 + 0.1) * output_size, (updated_pos[node][1] * 0.8 + 0.1) * output_size) for node in pos_dict}
    return updated_pos

def _get_node_color(node, evader_position, pursuer_positions, done):
    if node == evader_position:
        if done:
            return (255, 0, 0)
        else:
            return (0, 255, 0)
    elif node in pursuer_positions:
        return (0, 0, 255)
    else:
        return (255, 255, 255)

def render_pursuit_evasion_graph(graph: nx.Graph, pos: dict, evader_position, pursuer_positions, node_size=5, edge_width=1, done=False):
    return render_graph(graph, pos, node_size, edge_width, node_colors={
        node: _get_node_color(node, evader_position, pursuer_positions, done) for node in graph.nodes
    })

def render_graph(graph: nx.Graph, pos: dict, node_size, edge_width, node_colors=None):
    """
    Renders a graph using the given layout (`pos`), `node_size`, and `edge_width`. Node colors are determined
    with the `node_colors` parameter, which can be a (non-full) dictionary or `None`, with unspecified colors
    defaulting to white.
    """

    image = np.zeros((600, 600, 3), dtype=np.uint8)

    normalized_pos = normalize_pos_dict(pos)
    for edge in graph.edges():
        first, second = edge
        cv2.line(image, (int(normalized_pos[first][0]), int(normalized_pos[first][1])), (int(normalized_pos[second][0]), int(normalized_pos[second][1])), (255, 255, 255), edge_width)

    for node in graph.nodes:
        if type(node_colors) == dict:
            color = node_colors.get(node, (255, 255, 255))
        else:
            color = (255, 255, 255)
        
        x, y = normalized_pos[node]
        x = int(x)
        y = int(y)
        cv2.circle(image, (x, y), node_size, color, -1)

    return image

def draw_path(graph: nx.Graph, pos: dict, path: list, node_size: int, edge_width: int, custom_node_colors=None):
    # Draw original graph.
    # Start position is labeled in red, end position is labeled in green.
    rendered_graph = render_graph(graph, pos, node_size, edge_width, node_colors={
        path[0]: (0, 0, 255),
        path[-1]: (255, 0, 0),
        **(custom_node_colors or {}),
    })
    normalized_pos = normalize_pos_dict(pos)

    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        color = hsv_to_rgb(i / len(path), 0.8, 1)
        color = tuple([int(x * 255) for x in color])
        x1, y1 = normalized_pos[start_node]
        x2, y2 = normalized_pos[end_node]
        dx = x2 - x1
        dy = y2 - y1
        dx = dx / np.sqrt(dx ** 2 + dy ** 2)
        dy = dy / np.sqrt(dx ** 2 + dy ** 2)
        x1 += node_size * 1.2 * dx
        y1 += node_size * 1.2 * dy
        x2 -= node_size * 1.2 * dx
        y2 -= node_size * 1.2 * dy
        cv2.arrowedLine(rendered_graph, (int(x1), int(y1)), (int(x2), int(y2)), color, edge_width)

    return rendered_graph

def neighbors(y, x):
    return (
        (y + 1, x),
        (y, x + 1),
        (y - 1, x),
        (y, x - 1)
    )

def draw_edge_with_buffer(image, start, end, color, type, edge_thickness, arrow_buffer):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5
    direction_x = dx / distance
    direction_y = dy / distance
    start_x = start[0] + arrow_buffer * direction_x
    start_y = start[1] + arrow_buffer * direction_y
    end_x = end[0] - arrow_buffer * direction_x
    end_y = end[1] - arrow_buffer * direction_y
    start_x = int(start_x)
    start_y = int(start_y)
    end_x = int(end_x)
    end_y = int(end_y)
    if type == 'arrow':
        cv2.arrowedLine(image, (end_x, end_y), (start_x, start_y), color, edge_thickness, tipLength=0.5)
    elif type == 'line':
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, edge_thickness, lineType=cv2.LINE_AA)

def render_graph_v2(graph, pos, image_size, node_colors, edge_colors=(120, 120, 120), outline_colors=(50, 50, 50), node_size=5, edge_thickness=3, outline_thickness=3, arrows=[]):
    pos = normalize_pos_dict_square(pos, image_size)
    pos = {node: tuple(int(x) for x in pos[node]) for node in pos}

    image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    for node in graph.nodes:
        cv2.circle(image, pos[node], node_size, node_colors.get(node, (255, 255, 255)), -1)
        if type(outline_colors) == dict:
            cv2.circle(image, pos[node], node_size, outline_colors[node], outline_thickness, lineType=cv2.LINE_AA)
        else:
            cv2.circle(image, pos[node], node_size, outline_colors, outline_thickness, lineType=cv2.LINE_AA)

    edge_buffer = 1.4 * node_size
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            if type(edge_colors) == dict:
                color = edge_colors[(node, neighbor)]
            else:
                color = edge_colors
            if edge_thickness > 0:
                if (node, neighbor) in arrows:
                    draw_edge_with_buffer(image, pos[neighbor], pos[node], color, 'arrow', edge_thickness, edge_buffer)
                elif (neighbor, node) not in arrows:
                    draw_edge_with_buffer(image, pos[node], pos[neighbor], color, 'line', edge_thickness, edge_buffer)

    return image

def render_path_v2(graph, pos, image_size, path, background_color, custom_node_colors, node_size, edge_thickness, outline_thickness):
    node_colors = {
        node: background_color
        for node in graph.nodes
    }
    for i, node in enumerate(path):
        node_colors[node] = color_to_255(HSV(i / len(path))) # type: ignore
    node_colors.update(custom_node_colors)
    edge_colors = {
        (node, neighbor): (120, 120, 120)
        for node in graph.nodes
        for neighbor in graph.neighbors(node)
    }
    arrows = []
    for i, (initial_node, next_node) in enumerate(zip(path[:-1], path[1:])):
        edge_colors[initial_node, next_node] = color_to_255(HSV(i / len(path))) # type: ignore
        arrows.append((initial_node, next_node))
    outline_colors = {
        node: (50, 50, 50) if node in path else (120, 120, 120)
        for node in graph.nodes
    }
    return render_graph_v2(
        graph,
        pos,
        image_size,
        node_colors,
        edge_colors,
        outline_colors,
        node_size,
        edge_thickness,
        outline_thickness,
        arrows,
    )

def _render_video_render_path_section(graph, pos, path, custom_node_colors, node_size, image_size, i):
    trail = path[max(0, i - 5):i + 1]
    node_colors = {
        node: (255, 255, 255)
        for node in graph.nodes
    }
    for i in range(len(trail)):
        node_colors[trail[i]] = color_to_255(inferno(i / len(trail)))
    node_colors.update(custom_node_colors)

    return render_graph_v2(
        graph,
        pos,
        image_size,
        node_colors,
        node_size=node_size,
        edge_thickness=node_size // 10,
        outline_thickness=node_size // 5,
    )

def render_video(graph, pos, image_size, path, node_size, custom_node_colors, output_path, fps):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(tqdm.tqdm(executor.map(
            lambda i: _render_video_render_path_section(graph, pos, path, custom_node_colors, node_size, image_size, i),
            range(len(path)),
        )))

    # images = []
    # for i, node in tqdm.tqdm(enumerate(path), desc='Rendering video...'):
    #     trail = path[max(0, i - 5):i + 1]
    #     node_colors = {
    #         node: (255, 255, 255)
    #         for node in graph.nodes
    #     }
    #     for i in range(len(trail)):
    #         node_colors[trail[i]] = color_to_255(inferno(i / len(trail)))
    #     node_colors.update(custom_node_colors)

    #     image = render_graph_v2(
    #         graph,
    #         pos,
    #         image_size,
    #         node_colors,
    #         node_size=node_size,
    #         edge_thickness=node_size // 5,
    #         outline_thickness=node_size // 5,
    #     )
    #     images.append(image)
    
    # Create video

    import cv2

    height, width, layers = images[0].shape
    size = (width, height)

    # Use 8 fps
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for image in images:
        out.write(image)
    out.release()

def color_to_255(color):
    return tuple(int(x * 255) for x in color)
