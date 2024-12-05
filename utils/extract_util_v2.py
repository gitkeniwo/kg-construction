from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast, Set
from pydantic import create_model, BaseModel, Field
from langchain.schema.runnable import RunnableSequence
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from tqdm import tqdm

node_fields: Dict[str, Tuple[Any, Any]] = {
    "id": (
        str,
        Field(..., description="Name or human-readable unique identifier."),
    ),
    "type": (
        str,
        Field(..., description="The type or label of the node."),
    ),
}

NodeSchema = create_model("NodeSchema", **node_fields)

relationship_fields: Dict[str, Tuple[Any, Any]] = {
    "source": (
        str,
        Field(
            ...,
            description="Name or human-readable unique identifier of source node",
        ),
    ),
    "target": (
        str,
        Field(
            ...,
            description="Name or human-readable unique identifier of target node",
        ),
    ),
    "type": (
        str,
        Field(
            ...,
            description="The type of the relationship.",
        ),
    ),
}

RelationshipSchema = create_model("RelationshipSchema", **relationship_fields)


def convert_to_graph_documents(documents: List[Any], chain: RunnableSequence) -> List[GraphDocument]:
    """

    Parameters
    ----------
    documents
    chain : RunnableSequence

    Returns
    -------
    graph_documents : List[GraphDocument]

    """
    graph_documents = []

    # Enable tqdm progress bar
    with tqdm(total=len(documents), desc="Processing the documents...") as pbar:

        for i, doc in enumerate(documents):
            pbar.set_postfix({'Current chunk': doc.page_content[:10] + '...', 'Iteration': i + 1})

            result = chain.invoke({
                "text": doc.page_content,
            })

            nodes, relationships = parse_response(result)

            if not nodes or not relationships:
                continue

            graph_document = GraphDocument(nodes=nodes,
                                           relationships=relationships,
                                           source=doc)

            graph_documents.append(graph_document)

            pbar.update(1)

    return graph_documents


class BaseGraph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


def create_schema():
    """
    Create a KG schema for LLM structured output.
    Returns
    -------

    """

    class KGSchema(BaseGraph):
        """Represents a graph document consisting of nodes and relationships."""

        # nodes: Optional[List[Node]] = Field(description="List of nodes")
        # relationships: Optional[List[Relationship]] = Field(
        #     description="List of relationships"
        # )
        nodes: Optional[List[NodeSchema]] = Field(description="List of nodes")
        relationships: Optional[List[RelationshipSchema]] = Field(
            description="List of relationships"
        )

    return KGSchema


def parse_response(response: dict) -> Union[Tuple[List[Relationship], List[Node]], Tuple[None, None]]:
    """
    Parse the raw response from the LLM structured output.

    Parameters
    ----------
    response : dict
        The response from the model

    Returns
    -------
    relationships_list : List[Relationship]
        A list of relationships extracted from the text

    nodes_list : List[Node]
        A list of nodes extracted from the text

    """

    # error handling for missing raw response
    try:
        dictionary = response \
            .get('raw') \
            .response_metadata \
            .get('message')['tool_calls'][0]['function']['arguments']
    except Exception as e:
        print(f'Raw response message parsing error: {e}')
        return None, None

    try:
        node_dict = {node['id']: node['type'] for node in eval(dictionary['nodes'])}
    except Exception as e:
        print(f'Node dictionary parsing error: {e}')
        node_dict = {}

    node_set = set()

    relationships_list = []

    for relationship in eval(dictionary['relationships']):

        # error handling: if source or target is missing, then skip the relationship
        if (
                not relationship.get('source') or
                not relationship.get('target')
        ):
            continue

        source_node_tuple = (
            relationship['source'],
            node_dict.get(relationship['source'], 'node')
        )
        target_node_tuple = (
            relationship['target'],
            node_dict.get(relationship['target'], 'node')
        )

        node_set.add(source_node_tuple)
        node_set.add(target_node_tuple)

        relationships_list.append(Relationship(
            source=Node(
                id=source_node_tuple[0],
                type=source_node_tuple[1],
            ),
            target=Node(
                id=target_node_tuple[0],
                type=target_node_tuple[1],
            ),
            type=relationship.get('type', 'relationship')
        ))

    # print(relationships_list)

    nodes_list = [Node(id=node[0], type=node[1]) for node in node_set]

    # print(nodes_list)

    return nodes_list, relationships_list


# Graph Disambiguation / Merging Functionalities
def get_set_of_relationships(graph_documents: List[GraphDocument]) -> Set[Tuple[str, str, str]]:
    relationships = set()
    for graph_document in graph_documents:
        for relationship in graph_document.relationships:
            relationships.add((relationship.source.id, relationship.target.id, relationship.type))
    return relationships


def get_set_of_nodes(graph_documents: List[GraphDocument]) -> Set[Tuple[str, str]]:
    nodes = set()
    for graph_document in graph_documents:
        for node in graph_document.nodes:
            nodes.add((node.id, node.type))
    return nodes


def get_dict_of_nodes(set_of_nodes: Set[Tuple[str, str]]) -> Dict[str, Node]:
    return {node_id: Node(id=node_id, type=node_type) for node_id, node_type in set_of_nodes}


def merge_graphs(graph_documents: List[GraphDocument], source_document: Document) -> GraphDocument:
    """
    Resolve repeated nodes and relationships in the graph documents and merge them into a single graph document.

    Parameters
    ----------
    graph_documents
    source_document

    Returns
    -------

    """
    rel_set = get_set_of_relationships(graph_documents)
    node_set = get_set_of_nodes(graph_documents)

    node_dict = get_dict_of_nodes(node_set)

    # create Node, Relationship objects from the sets
    merged_nodes = list(node_dict.values())

    merged_relationships = [
        Relationship(
            source=node_dict[source_id],
            target=node_dict[target_id],
            type=relationship_type
        ) for source_id, target_id, relationship_type in rel_set
    ]

    return GraphDocument(nodes=merged_nodes,
                         relationships=merged_relationships,
                         source=source_document)