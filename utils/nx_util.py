import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


def draw_kg(G: nx.Graph,
            with_labels=True,
            k=2, figsize=(5, 5), node_size=500, width=1,
            file_name='./output/pics/example.png',
            color_palette="hsv",
            font_size=10,
            **kwargs):
    """
    Draw a Knowledge Graph using NetworkX and Matplotlib.

    :param G:  Knowledge Graph
    :param with_labels:  Whether to display node labels
    :param k:  Spring layout parameter
    :param figsize:  Figure size
    :param node_size:  Node size
    :param width:  Edge width
    :param file_name:  File name to save the plot
    :param font_size:  Font size for node labels
    :param kwargs:  Additional keyword arguments for nx.draw_networkx
    :return:  Figure and axis objects
    """

    # configure color palette
    node_types = list(set([G.nodes[node]['type'] for node in G.nodes()]))
    palette = sns.color_palette(color_palette, len(node_types))
    category_to_color = {cat: palette[i] for i, cat in enumerate(node_types)}
    node_colors = [category_to_color[G.nodes[node]["type"]] for node in G.nodes]
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=category)
                       for category, color in category_to_color.items()]

    # print(category_to_color)

    # plot settings
    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)

    options = {
        "node_size": node_size,
        "linewidths": 0.2,
        "width": width,
        **kwargs,
    }

    pos = nx.spring_layout(G, k=k, seed=42)
    # pos = nx.planar_layout(G, scale=2)

    nx.draw_networkx(G,
                     pos=pos,
                     with_labels=with_labels,
                     node_color=node_colors,
                     **options,
                     ax=ax)

    nx.draw_networkx_edge_labels(G,
                                 pos=pos,
                                 edge_labels=nx.get_edge_attributes(G, 'relation'),
                                 ax=ax)

    plt.legend(handles=legend_elements, loc="best", title="Node Types", fontsize=8)
    plt.savefig(file_name)
    plt.show()

    return fig, ax


def subgraph_by_entity_type(G, node_type):
    """
    Extract subgraph by node type
    """

    # if node_type is not a list
    if not isinstance(node_type, list):
        node_type = [node_type]

    nodes_of_type_X = [node for node, attr in G.nodes(data=True) if attr.get("type") in node_type]

    subgraph = G.subgraph(nodes_of_type_X)

    return subgraph



