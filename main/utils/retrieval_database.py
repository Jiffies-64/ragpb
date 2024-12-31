import math
import os
import sqlite3
from typing import List

import openai
import torch
import langchain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from chardet.universaldetector import UniversalDetector
from langchain_core.documents import Document
from tqdm import tqdm

from main.prompt.extract_claims import CLAIMS_EXTRACTION_TEMPLATE, CLAIMS_EXTRACTION_EXAMPLES, \
    CLAIMS_EXTRACTION_EXAMPLES_BY_PERSON, CLAIMS_EXTRACTION_TEMPLATE_BY_PERSON
from main.utils.constant import root_dir
from main.utils.llm_factory import LLMManager
from main.utils.parse_json import parse_resp_to_json
from main.utils.uuid_util import generate_4_digit_uuid


def find_all_file(path: str) -> List[str]:
    """
    Return the list of all files in a folder.
    :param path: The path of the folder.
    :return: A list containing the paths of all files in the folder.
    """
    for root, ds, fs in os.walk(path):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_encoding_of_file(path: str) -> str:
    """
    Return the encoding of a file.
    """
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        data = file.readlines()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']


def get_embed_model(encoder_model_name: str,
                    device: str = 'cpu',
                    retrival_database_batch_size: int = 256) -> OpenAIEmbeddings:
    """
    Get the embedding model.
    You can also code for other embedding models.
    :param encoder_model_name: Name of encoder model. Options:
        open-ai: OpenAI Embeddings
        all-MiniLM-L6-v2: default Embedding method
        bge-large-en-v1.5: bge-large-en-v1.5 from BAAI
        e5-base-v2: e5-base-v2 from intfloat
    :param device: cpu or gpu if available.
    :param retrival_database_batch_size: Batch size.
    :return: The embedding model.
    """
    if encoder_model_name == 'open-ai':
        embed_model = OpenAIEmbeddings(
            openai_api_base="https://chatapi.onechats.top/v1",
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif encoder_model_name == 'all-MiniLM-L6-v2':
        embed_model = HuggingFaceEmbeddings(
            model_name=encoder_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size},
        )
    elif encoder_model_name == 'bge-large-en-v1.5':
        embed_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size}
        )
    elif encoder_model_name == 'e5-base-v2':
        embed_model = HuggingFaceEmbeddings(
            model_name='intfloat/e5-base-v2',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size}
        )
    else:
        try:
            embed_model = HuggingFaceEmbeddings(
                model_name=encoder_model_name,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size},
            )
        except encoder_model_name:
            raise Exception(f"Encoder {encoder_model_name} not found, please check.")
    return embed_model


def sim_to_dissimilar(x):
    """
    Map a non-negative number (can be positive infinity) to the interval 0-1, where 0 maps to 1 and positive infinity maps to 0.
    """
    if x == 0:
        return 1
    elif x == float('inf'):
        return 0
    else:
        return 1 / (1 + x)


def split_docs_to_claims(db_path, raw_text_list, task_id, group_by_person=False):
    """
    Split documents into claims and store them in the database.
    :param db_path: The path to the database.
    :param raw_text_list: The list of raw text documents.
    :param task_id: The task ID.
    :param group_by_person: Whether to group by person.
    """
    conn = sqlite3.connect(db_path)

    if group_by_person:
        icl_samples = "\n".join(
            f"Document: {example['Document']}\nClaims:\n{example['Claims']}"
            for example in CLAIMS_EXTRACTION_EXAMPLES_BY_PERSON
        )
    else:
        icl_samples = "\n".join(
            f"Document: {example['Document']}\nClaims:\n" + "<SEP>".join(example['Claims'])
            for example in CLAIMS_EXTRACTION_EXAMPLES
        )

    cursor = conn.cursor()
    # Record the processed document index, initialized to 0, can be restored from the database if there is a previous record
    processed_index = 0
    try:
        # Try to read the processed index value from the database
        cursor.execute(f"SELECT processed_index FROM processing_status WHERE id = '{task_id}'")
        result = cursor.fetchone()
        if result:
            processed_index = result[0]
    except sqlite3.OperationalError:
        # If the table does not exist, it is the first run, no need to process, continue with the initial value 0
        pass

    if group_by_person:
        for index, ds in tqdm(enumerate(raw_text_list)):
            if index < processed_index:
                continue
            key = ds['id']
            value = ds['content']
            prompt = CLAIMS_EXTRACTION_TEMPLATE_BY_PERSON.format(icl_samples=icl_samples, document=value)
            llm = LLMManager(LLMManager.ModelEnum.GPT4o)
            response = llm.get_llm_output(prompt)
            response_dict = parse_resp_to_json(response)
            for name, claims in response_dict.items():
                for claim in claims:
                    cursor.execute("INSERT INTO claims (task_id, id, claim) VALUES (?,?,?)",
                                   (task_id, f"{key}<SEP>{name}", claim))
            # Update the processed index value in the database
            cursor.execute("INSERT OR REPLACE INTO processing_status (id, processed_index) VALUES (?,?)",
                           (task_id, index + 1))
            conn.commit()
    else:
        for index, ds in enumerate(raw_text_list):
            if index < processed_index:
                continue
            key = ds['id']
            value = ds['content']
            prompt = CLAIMS_EXTRACTION_TEMPLATE.format(icl_samples=icl_samples, document=value)
            llm = LLMManager(LLMManager.ModelEnum.GPT4o)
            response = llm.get_llm_output(prompt)
            claims = response.split("<SEP>")
            for claim in claims:
                cursor.execute("INSERT INTO claims (task_id, id, claim) VALUES (?,?,?)", (task_id, key, claim))
            # Update the processed index value in the database
            cursor.execute("INSERT OR REPLACE INTO processing_status (id, processed_index) VALUES (?,?)",
                           (task_id, index + 1))
            conn.commit()


def read_split_result_from_db(db_path, task_id):
    """
    Read the split result from the database.
    :param db_path: The path to the database.
    :param task_id: The task ID.
    :return: A dictionary containing the split result.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    result_dict = {}
    try:
        # Query the claims table to get all data
        cursor.execute(f"SELECT id, claim FROM claims WHERE task_id = '{task_id}'")
        rows = cursor.fetchall()
        for row in rows:
            doc_id = row[0]
            claim_text = row[1]
            if doc_id not in result_dict:
                result_dict[doc_id] = []
            result_dict[doc_id].append(claim_text)
    except sqlite3.Error as e:
        print(f"Error occurred when reading the database: {e}")
    finally:
        conn.close()

    return result_dict


def construct_retrieval_database(
        dataset_name: str,
        database_id: str,
        document_list: List[Document],
        encoder_model_name: str = 'all-MiniLM-L6-v2',
        retrival_database_batch_size: int = 256,
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
        mode: str = 'rebuild'  # Default is to rebuild (overwrite existing data)
) -> 'langchain.vectorstores.chroma.Chroma':
    """
    Construct or load a retrieval database.
    :param dataset_name: The name of the dataset.
    :param database_id: The ID of the database.
    :param document_list: The list of documents.
    :param encoder_model_name: The name of the encoder model.
    :param retrival_database_batch_size: The batch size for retrieval.
    :param chunk_size: The size of chunks.
    :param chunk_overlap: The size of chunk overlap.
    :param mode: The mode of construction ('rebuild', 'load', 'merge').
    :return: The constructed Chroma vector store.
    """
    vector_store_path = os.path.join(root_dir, 'store', dataset_name, encoder_model_name, database_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(encoder_model_name, device, retrival_database_batch_size)
    if mode == 'rebuild':
        # Check if the path exists, if so, delete the content under the path (equivalent to overwriting existing data)
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
        retrieval_database = Chroma.from_documents(
            documents=document_list, embedding=embed_model, persist_directory=vector_store_path
        )
    elif mode == 'load':
        if os.path.exists(vector_store_path):
            try:
                retrieval_database = Chroma(
                    embedding_function=embed_model,
                    persist_directory=vector_store_path
                )
                return retrieval_database
            except Exception as e:
                print(f"Failed to load existing database, will rebuild it. Error message: {str(e)}")
        retrieval_database = Chroma.from_documents(
            documents=document_list, embedding=embed_model, persist_directory=vector_store_path
        )
    elif mode == 'merge':
        if os.path.exists(vector_store_path):
            existing_database = Chroma(
                embedding_function=embed_model,
                persist_directory=vector_store_path
            )
            existing_docs = existing_database.get_documents()
            all_docs = existing_docs + document_list
        else:
            all_docs = document_list
        retrieval_database = Chroma.from_documents(
            documents=all_docs, embedding=embed_model, persist_directory=vector_store_path
        )
    else:
        raise ValueError("Invalid mode specified. Mode should be one of 'rebuild', 'load' or 'merge'.")
    return retrieval_database


def construct_split_retrieval_database(dataset_name, split_result: dict[str, list[str]]):
    """
    Construct a split retrieval database with adjusted weights based on similarity and document length.
    Similar document number: For each document, consider the number of occurrences in the list of similar documents. Fewer similar documents indicate stronger saliency.
    Similarity score weighting: Take the average or weighted average of similarity scores (considering document length). Lower overall similarity scores indicate weaker associations and stronger saliency.
    Length penalty (optional): Reduce the weight of shorter documents to avoid low similarity due to less content (noise).
    Composite score: Calculate saliency as a function of the number of similar documents and similarity scores, and normalize appropriately.
    """
    # Define constants
    MIN_K4WEIGHT = 8  # Minimum number of similar documents to select during secondary construction
    MAX_K4WEIGHT = 256  # Maximum number of similar documents to select during secondary construction
    SIM_THRESHOLD = 0.6  # Only consider statements with similarity greater than 0.6 during secondary construction

    # First, build the vector database for the entire dataset, metadata only contains id (here identified by name)
    documents = []
    for key, claims in split_result.items():
        metadata = {"key": key}
        for claim in claims:
            documents.append(Document(page_content=claim, metadata=metadata))
    rdb_tmp = construct_retrieval_database(
        dataset_name=dataset_name,
        database_id="tmp",
        document_list=documents,
        mode="rebuild"
    )

    # Secondary construction, aiming to obtain weights. The fewer the number of similar contents and the lower the similarity, the stronger the saliency, and higher weights are required.
    documents = []
    for key, claims in split_result.items():
        k4weight = MIN_K4WEIGHT
        for claim in claims:
            while True:
                # Get the k most similar documents and their scores
                similar_docs_with_scores = rdb_tmp.similarity_search_with_score(claim, k=k4weight)
                similar_docs_with_scores = sorted(similar_docs_with_scores, key=lambda x: x[1], reverse=True)

                # Check if the parameters used in this retrieval meet the requirements
                num_similar_docs = len(similar_docs_with_scores)
                last_similarity = similar_docs_with_scores[-1][1]
                if (
                        num_similar_docs != k4weight  # The actual number retrieved is not equal to the expected number (only possibly less), i.e., the vector database content limit is reached
                        or num_similar_docs >= MAX_K4WEIGHT  # The retrieved number is greater than or equal to the specified maximum
                        or last_similarity <= SIM_THRESHOLD
                # The retrieved similarity is less than or equal to the minimum
                ):
                    break
                else:
                    k4weight *= 2
            print("[DEBUG]num_similar_docs = {}, last_similarity = {}".format(num_similar_docs, last_similarity))

            # Count the number of similar documents and the sum of similarities
            total_dissimilar_score = count = 0
            for document, score in similar_docs_with_scores:
                if score < SIM_THRESHOLD:
                    break
                doc_length = len(document.page_content)  # Document length
                dis_sim = sim_to_dissimilar(
                    score / max(math.log10(doc_length), 1))  # Use dissimilarity to represent saliency
                total_dissimilar_score += dis_sim
                count += 1

            # Saliency weight calculation
            avg_dissimilar = max(total_dissimilar_score, 1) / max(count, 1)  # Average dissimilarity

            # Add the document and record the saliency weight
            documents.append(Document(
                page_content=claim,
                metadata={"key": key, "weight": avg_dissimilar}
            ))
    return construct_retrieval_database(
        dataset_name=dataset_name,
        database_id="split-text",
        document_list=documents,
        mode="rebuild"
    )


def load_retrieval_database_from_parameter(
        dataset_name: str,
        database_id: str,
        encoder_model_name: str = 'all-MiniLM-L6-v2',
        retrival_database_batch_size: int = 512
) -> 'langchain.vectorstores.chroma.Chroma':
    """
    Load the retrieval database by given parameters.
    :param dataset_name: The name of the dataset.
    :param database_id: The ID of the database.
    :param encoder_model_name: The name of the encoder model.
    :param retrival_database_batch_size: The batch size of the retrieval database.
    :return: The loaded Chroma vector store.
    """
    print("Starting load_retrieval_database_from_parameter...")
    vector_store_path = os.path.join(root_dir, 'store', dataset_name, encoder_model_name, database_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = get_embed_model(
        encoder_model_name, device, retrival_database_batch_size
    )
    print("Staring to load the retrieval database...")
    retrieval_database = Chroma(
        embedding_function=embed_model,
        persist_directory=vector_store_path
    )
    print(
        "Succeeded in loading the retrieval database! The file had been loaded into {store_path}"
        .format(store_path=vector_store_path)
    )
    return retrieval_database
