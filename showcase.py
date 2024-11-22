import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils.nx_util import draw_kg, subgraph_by_entity_type

# Load the graph data
with open('./output/relations/chunk-level-relations.pickle', 'rb') as file:
    cl_relations = pickle.load(file)

with open('./output/entities/global-entities.txt', 'r') as file:
    global_entities = eval(file.read())

with open('./output/relations/relations-entity-type.pickle', 'rb') as file:
    entity_types_mapping = pickle.load(file)

KG = nx.DiGraph()

for relation in cl_relations:
    try:
        KG.add_edge(relation[0], relation[1], relation=relation[2], relation_type=relation[3])
    except IndexError as e:
        print(f'ImportRelationsError: item: {relation} => error: {e}')

for node in KG.nodes():
    KG.nodes[node]['type'] = entity_types_mapping.get(node, 'Miscellaneous')


# Display the graph in Streamlit
available_labels = list(set(entity_types_mapping.values()))


# Sidebar for multi-select
st.sidebar.header("Filter options")
selected_labels = st.sidebar.multiselect(
    "Select entity labels to display:",
    options=available_labels,
    default=available_labels,  # Default selections
)

# set the book name, A Christmas Carol,  in italics
st.title("Knowledge Graph Visualization")
st.write("This is a visualization of the knowledge graph extracted from Charles Dickens' A Christmas Carol.")
st.write("The graph is based on the relations between entities in the text.")
st.write("You can filter the graph by selecting labels in the sidebar.")


if selected_labels:

    subgraph = subgraph_by_entity_type(KG, selected_labels)

    fig, ax = draw_kg(subgraph,
        k=1e-8,
        figsize=(25, 25),
        node_size=500,
        file_name='./output/pics/kg-networkx-example.png',
        color_palette="hsv",
        font_size=5,
        )

    ax.set_title("Selected Labels:" + ", ".join(selected_labels), fontsize=20)

    ax.legend()
    st.pyplot(fig)
else:
    st.write("No labels selected. Please choose labels from the sidebar.")



