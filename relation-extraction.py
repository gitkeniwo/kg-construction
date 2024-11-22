#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 1 Text Chunking
# 
# For the sake of model max token limits, we need to process the text chapter by chapter.
# 
# Since we cannot fit the whole document into the model, we need to split the text into smaller chunks. We can use the `CharacterTextSplitter` to split the text into smaller chunks.
# 
# `chunk_overlap` sets the overlapping tokens between the chunks, which serves as a sliding window that moves over the text, avoiding splitting the text in the middle of a sentence.

# In[13]:


# Text chunking

from utils.extraction_util import extract_entities

from tqdm import tqdm
import yaml
import pickle

from langchain_community.tools.file_management.write import WriteFileTool
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

book_file_paths = ['./data/a-xmax-carol-stave' + str(i + 1) + '.txt' for i in range(5)]

texts = []
for book_file_path in book_file_paths:
    with open(book_file_path, 'r') as file:
        texts.append(file.read())

# Initialize the splitter
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=2048,
    chunk_overlap=256,
)

text = texts[0]

# Create RunnableLambda for the splitter
splitter_lambda = lambda text: splitter.create_documents([text])
splitter_runnable = RunnableLambda(splitter_lambda)

chunks = splitter.create_documents([text])

chunks_with_id = [(i, chunk) for i, chunk in enumerate(chunks)]

# # Iterative Entity Extraction
# 
# Entity extraction is a two-step process:
# - Chunk-level Entity Extraction (CLEE)
# - Entity Aggregation (EA)
# 
# For the sake of model max token limits, we need to process the text chapter by chapter.
# 
# The process is shown below, and is wrapped in another python module `extraction.py`.

# ## Chunk-level Entity Extraction
# 
# Next, we build a simple pipeline for Chunk-level Entity Extraction (CLEE) using the Ollama model.

# In[17]:


# use ollama llama3.1:8b for entity extraction

# read prompt files from yaml
with open('./prompts/entity.yaml', 'r') as file:
    entity_yaml = yaml.safe_load(file)

# Initialize the ollama model    
llama3_1_params = entity_yaml['chat_model_params']['llama3.1']

llama3_1 = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    top_k=10,
    top_p=0.6,
    num_predict=2048,
    base_url="http://127.0.0.1:11434",
)

# chunk-level entity extraction (CLEE)
clee_template = ChatPromptTemplate.from_messages([
    ("system", entity_yaml['system']),
    ("human", entity_yaml['clee']),
])

clee_chain = clee_template | llama3_1 | StrOutputParser()

# Next, we define an iterative function that runs the CLEE on each chunk and caches the results.

# In[5]:


# Define Iterative RunnableLambda of Chunk_Level Entity Extraction


# ## Stave-level Entity Aggregation
# 
# Next, we define a template for Entity Aggregation (EA) and run it on the cached chunk-level entities for each stave.

# ## The Result for All Staves
# 
# The above process only for the first stave. The logic can be applied to the rest of the staves and is wrapper in another python module `extraction_util.py`.
# We can now run the process for all the staves.

# In[6]:


# read prompt files from yaml
with open('./prompts/entity.yaml', 'r') as file:
    entity_yaml = yaml.safe_load(file)

stave_entities = []
for i, text in enumerate(texts):
    stave_entities.append(extract_entities(text=text, entity_yaml=entity_yaml, stave_num=i + 1))

print("Stave entities extracted successfully!")

# ## Aggregate over the Stave-level Entities
# 
# Next, we define a template for Stave-level Entity Aggregation and run it on the cached chunk-level entities for each stave.
# 
# We also run by chunks iteratively, provide the model with the previously extracted entities to maintain the context.

# In[8]:


llama3_1_sle = ChatOllama(
    model="llama3.1:8b",
    temperature=0.2,
    top_k=10,
    top_p=0.6,
    num_predict=4096,
)

# The entity list is still too long for LLM inputs. So, let's split the list into smaller chunks, and feed them to the model iteratively, together with the model output context from the previous iteration.

# In[9]:

with open('./prompts/entity.yaml', 'r') as file:
    entity_yaml = yaml.safe_load(file)

stave_aggregation_template = ChatPromptTemplate.from_messages([
    ("system", entity_yaml['system']),
    ("human", entity_yaml['stave-aggregation']),
])

stave_aggregation_chain = stave_aggregation_template | llama3_1_sle | StrOutputParser()

previous_entities = ""

for stave in tqdm(stave_entities):
    global_entities = stave_aggregation_chain.invoke({
        "entity_type": entity_yaml["entity_types"],
        "entities": stave,
        "prev_entities": previous_entities,
    })

    previous_entities = global_entities

print("Global entities extracted successfully!")

# cache the global entities
WriteFileTool().invoke({
    "file_path": "./output/global-entities.txt",
    "text": global_entities,
})

# or using pickle

with open('output/entities/global-entities.pickle', 'wb') as file:
    pickle.dump(global_entities, file)

# # 3 Relation Extraction
# 
# First, we ask the llm to provide us with a list of possible relation types.

# In[67]:


llama3_1_relation_type = ChatOllama(
    model="llama3.1:8b",
    temperature=0.8,
    top_k=60,
    top_p=0.9,
    num_predict=2048,
    base_url="http://127.0.0.1:11434",
)

relation_type_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are to assist the user to construct a knowledge graph for a novel. "),
    ("human", """
The user is working on extracting relations from a book to build a knowledge graph. You goal is to provide a list of possible relation types that are commonly present in knowledge graphs for novels.
Please reply in the following format:
["relation_type1", "relation_type2", "relation_type3", ...]
"""),
])

relation_type_chain = relation_type_prompt_template | llama3_1_relation_type | StrOutputParser()

relation_type_chain.invoke({})

# ## 3.1 Chunk-level Relation Extraction
# 
# Let's run Relation Extraction (RE) on each chunk with the help of extracted global entities.
# 
# Here's the a test run on the first chunk.

# In[23]:


with open('./output/entities/global-entities.txt', 'r') as file:
    global_entities = eval(file.read())

with open("./prompts/relation.yaml", 'r') as file:
    relation_yaml = yaml.safe_load(file)

# chunk-level relation extraction
cl_relation_extraction_template = ChatPromptTemplate.from_messages([
    ("system", relation_yaml['system']),
    ("human", relation_yaml['chunk-relation-extraction-basic']),
])

cl_relation_extraction_chain = cl_relation_extraction_template | llama3_1 | StrOutputParser()

# ## Iterative Chunk-Level Relation Extraction

# In[25]:


whole_book = "./data/a-xmas-carol-body.txt"

with open(whole_book, 'r') as file:
    whole_book_text = file.read()

# Initialize the splitter
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=2048,
    chunk_overlap=256,
)

chunks_whole_book = splitter.create_documents([whole_book_text])

# chunk-level relation extraction
cl_relation_extraction_template = ChatPromptTemplate.from_messages([
    ("system", relation_yaml['system']),
    ("human", relation_yaml['chunk-relation-extraction-basic']),
])

cl_relation_extraction_chain = cl_relation_extraction_template | llama3_1 | StrOutputParser()


def iter_extract_relations(chunks, chain=cl_relation_extraction_chain):
    chunk_level_relations = []

    # Iterate over the chunks and extract relations
    with tqdm(total=len(chunks), desc="Processing the chunks ...") as pbar:
        for i, chunk in enumerate(chunks):
            pbar.set_postfix({'Current chunk': chunk.page_content[:10] + '...', 'Iteration': i + 1})

            chunk_level_relations.append(chain.invoke({
                "global_entities": global_entities,
                "current_chunk": chunk.page_content,
                "relation_types": relation_yaml["relation-types"],
            }))
            pbar.update(1)

    # Cache the chunk-level relations for the stave
    WriteFileTool().invoke({
        "file_path": "./output/relations/chunk-level-relations.txt",
        "text": str(chunk_level_relations),
    })

    return chunk_level_relations


chunk_level_relations = iter_extract_relations(chunks_whole_book)

# In[31]:

# output processing and cache the results
chunk_level_relations_eval = []

for relations in chunk_level_relations:
    try:
        chunk_level_relations_eval += eval(relations)
    except Exception as e:
        print(e)


with open('output/relations/chunk-level-relations.pickle', 'wb') as file:
    pickle.dump(chunk_level_relations_eval, file)

# ## 3.2 Entity Type Resolution
# 
# The entities in the relations extracted from the last step does not really match the entities we extracted separately before.
# 
# We need to resolve the entity types for them using LLMs again. 
# 
# The result will be a hashmap (python dictionary) with the entity as the key and the entity type as the value.

# In[39]:


with open('./output/relations/chunk-level-relations.pickle', 'rb') as file:
    cl_relations = pickle.load(file)

# In[106]:


entities_in_extracted_relations = []

for cl_relation in cl_relations:
    entities_in_extracted_relations.append(cl_relation[0])
    entities_in_extracted_relations.append(cl_relation[1])

entities_in_extracted_relations = list(set(entities_in_extracted_relations))

# Next, build another pipeline for entity type resolution.

# In[113]:


# resolve entity types

relation_yaml = yaml.safe_load(open('./prompts/relation.yaml', 'r'))

assign_entity_types_template = ChatPromptTemplate.from_messages([
    ("system", relation_yaml['system']),
    ("human", relation_yaml['assign-entity-types']),
])

llama3_1_assign_entity_type = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,
    top_k=20,
    top_p=0.6,
    num_predict=4096,
    base_url="http://127.0.0.1:11434",
)

assign_entity_types_chain = assign_entity_types_template | llama3_1_assign_entity_type | StrOutputParser()

entity_types_for_relations = []

for entity in tqdm(entities_in_extracted_relations):
    entity_types_for_relations.append(assign_entity_types_chain.invoke({
        "entity": entity,
        "entity_types": entity_yaml["entity_types"],
    }))

# In[123]:


# error handling: if the entity type does not belong to the list, then assign the entity type as 'Miscellaneous'

final_entity_types_list = []
valid_answer = 0

for item in entity_types_for_relations:
    try:
        if item in entity_yaml["entity_types"]:
            final_entity_types_list.append(item)
            valid_answer += 1
        else:
            final_entity_types_list.append("Miscellaneous")

    except Exception as e:
        print(e)
        final_entity_types_list.append("Miscellaneous")

print(f'{valid_answer} / {entity_types_for_relations.__len__()} valid entity types found.')

entity_types_mapping = dict(zip(entities_in_extracted_relations, final_entity_types_list))

# dump to pickle

with open('./output/relations/relations-entity-type.pickle', 'wb') as file:
    pickle.dump(entity_types_mapping, file)
