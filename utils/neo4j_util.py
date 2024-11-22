from neo4j import GraphDatabase
import networkx as nx


class Neo4jGraphImporter:
    """
    Import a NetworkX graph into a Neo4j database.

    Parameters:
    - uri: URI of the Neo4j database
    - user: Username for the Neo4j database
    - password: Password for the Neo4j database

    ```
    # Example: Create a NetworkX graph
    G = nx.DiGraph()
    G.add_node(1, type="Person", name="Alice")
    G.add_node(2, type="Person", name="Bob")
    G.add_edge(1, 2, label="KNOWS", since=2021)

    # Instantiate the importer and load the graph into Neo4j
    neo4j_importer = Neo4jGraphImporter("bolt://localhost:7687", "neo4j", "password")
    neo4j_importer.clear_database()  # Optional: Clear the database
    neo4j_importer.import_graph(G)
    neo4j_importer.close()
    ```
    """

    def __init__(self, uri, user, password, db_name="neo4j"):
        """
        Initialize the Neo4jGraphImporter with connection details.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db_name = db_name

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()

    def clear_database(self):
        """
        Clear the entire database (delete all nodes and relationships).
        """
        with self.driver.session(database=self.db_name) as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.execute_write(self._remove_all_labels)
            print("Database cleared.")

    def drop_all_constraints(self):
        """
        Drop all constraints in the database.
        """
        with self.driver.session(database=self.db_name) as session:
            # Get the list of constraints
            result = session.run("SHOW CONSTRAINTS")

            # Iterate over the constraints and drop each one by name
            for record in result:
                constraint_name = record['name']
                # Drop the constraint
                session.run(f'DROP CONSTRAINT `{constraint_name}`')

            print("All constraints dropped.")

    def import_graph(self, nx_graph):
        """
        Import a NetworkX graph into the Neo4j database.
        """
        with self.driver.session(database=self.db_name) as session:
            session.execute_write(self._add_nodes, nx_graph)
            session.execute_write(self._add_relationships, nx_graph)

    @staticmethod
    def _add_nodes(tx, graph):
        """
        Add nodes from a NetworkX graph to the database.
        """
        for node_id, attributes in graph.nodes(data=True):
            node_type = attributes.get("type", "DefaultType")
            query = f"""
            MERGE (n:{node_type} {{id: $id}})
            SET n += $properties
            """
            tx.run(query, id=node_id, properties=attributes)

    # @staticmethod
    # def _add_relationships(tx, graph):
    #     """
    #     Add relationships from a NetworkX graph to the database.
    #     """
    #     for source, target, attributes in graph.edges(data=True):
    #         label = attributes.get("relation", "RELATED")
    #         query = f"""
    #         MATCH (a {{id: $source_id}})
    #         MATCH (b {{id: $target_id}})
    #         MERGE (a)-[r:`{label}`]->(b)
    #         SET r += $properties
    #         """
    #         tx.run(query, source_id=source, target_id=target, properties=attributes)

    @staticmethod
    def _add_relationships(tx, graph):
        """
        Add relationships from a NetworkX graph to the database.
        """
        for source, target, attributes in graph.edges(data=True):
            label = attributes.get("relation", "RELATED").replace(" ", "_")
            relation_type = attributes.get("relation-type", "default-type")
            query = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            MERGE (a)-[r:`{label}`]->(b)
            SET r += $properties, r.relation_type = $relation_type
            """
            tx.run(query, source_id=source, target_id=target, properties=attributes, relation_type=relation_type)

    @staticmethod
    def _remove_all_labels(tx):
        query = """
        MATCH (n)
        WITH n, labels(n) AS labels
        FOREACH (label IN labels | REMOVE n:`label`)
        """
        tx.run(query)
