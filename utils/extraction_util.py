from tqdm import tqdm
from langchain_community.tools.file_management.write import WriteFileTool
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pickle


def extract_entities(text, entity_yaml, stave_num=1):
    splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size=2048,
        chunk_overlap=256,
    )

    # Create RunnableLambda for the splitter
    splitter_lambda = lambda text: splitter.create_documents([text])
    splitter_runnable = RunnableLambda(splitter_lambda)

    chunks = splitter.create_documents([text])

    chunks_with_id = [(i, chunk) for i, chunk in enumerate(chunks)]

    # Initialize the ollama model
    llama3_1_params = entity_yaml['chat_model_params']['llama3.1']

    llama3_1 = ChatOllama(
        model=llama3_1_params['model'],
        temperature=llama3_1_params['temperature'],
        top_k=llama3_1_params['top_k'],
        top_p=llama3_1_params['top_p'],
        num_predict=llama3_1_params['num_predict'],
        base_url=llama3_1_params['base_url'],
    )

    # CLEE: Define Iterative RunnableLambda of Chunk_Level Entity Extraction
    def iterate_over_chunks_and_cache(chunks):
        # chunk-level entity extraction (CLEE)
        clee_template = ChatPromptTemplate.from_messages([
            ("system", entity_yaml['system']),
            ("human", entity_yaml['clee']),
        ])

        clee_chain = clee_template | llama3_1 | StrOutputParser()

        chunk_level_entities = []

        # Iterate over the chunks and extract entities
        with tqdm(total=len(chunks), desc="Processing the chunks of stave"+str(stave_num)+"...") as pbar:
            for i, chunk in enumerate(chunks):
                pbar.set_postfix({'Current chunk': chunk.page_content[:10] + '...', 'Iteration': i + 1})
                chunk_level_entities.append(clee_chain.invoke({
                    "text": chunk.page_content,
                    "entity_type": entity_yaml["entity_types"],
                }))
                pbar.update(1)

        # Cache the chunk-level entities for the stave
        WriteFileTool().invoke({
            "file_path": "./output/entities/chunk-level-entities-stave"+str(stave_num)+".txt",
            "text": str(chunk_level_entities),
        })

        # or using pickle
        with open("./output/entities/chunk-level-entities-stave"+str(stave_num)+".pickle", "wb") as f:
            pickle.dump(chunk_level_entities, f)

        return {
            "entities": chunk_level_entities,
            "entity_type": entity_yaml["entity_types"],
        }

    clee_iter = RunnableLambda(iterate_over_chunks_and_cache)

    # Entity-aggregation
    # params: entity-type, entities
    ea_template = ChatPromptTemplate.from_messages([
            ("system", entity_yaml['system']),
            ("human", entity_yaml['ea']),
        ])

    ea_chain = splitter_runnable | clee_iter | ea_template | llama3_1 | StrOutputParser()

    stave_entities = ea_chain.invoke(text)

    # Cache the stave-level entities
    WriteFileTool().invoke({
        "file_path": "./output/entities/stave"+str(stave_num)+"-entities.txt",
        "text": str(stave_entities),
    })

    # or using pickle
    with open("./output/entities/stave"+str(stave_num)+"-entities.pickle", "wb") as f:
        pickle.dump(stave_entities, f)

    return stave_entities
