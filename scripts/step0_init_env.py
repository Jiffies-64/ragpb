import argparse
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

from main.dao.database import init_temp_database_for_claims
from main.dataset.__all__ import DATASETS_NAME_TO_DATA_LIST
from main.utils.constant import root_dir
from main.utils.retrieval_database import construct_retrieval_database, split_docs_to_claims, read_split_result_from_db, \
    construct_split_retrieval_database
from main.utils.uuid_util import generate_4_digit_uuid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="You can choose from the following datasets: chatdoctor-plus, legal-plus. "
             "This parameter specifies the dataset(s) to be used in the process.",
        required=False,
        default=["legal-plus", ],  # "chatdoctor-plus",
        type=list
    )
    parser.add_argument(
        "--limit",
        help="Maximum number of records loaded into the dataset",
        required=False,
        default=-1,
        type=int
    )
    return parser.parse_args()


if __name__ == '__main__':

    # load or define variables
    args = parse_args()
    encoder_model = 'all-MiniLM-L6-v2'
    # encoder_model = 'open-ai'
    load_dotenv(os.path.join(root_dir, '.env'))

    # init system database for logging etc.
    db_path = os.path.join(root_dir, 'store', 'system.sqlite')
    if not os.path.exists(db_path):
        init_temp_database_for_claims(db_path)

    # for each dataset
    for dataset_name in args.dataset:
        raw_text_list = DATASETS_NAME_TO_DATA_LIST[dataset_name].read_content(args.limit)

        raw_text_document_list = [
            Document(
                page_content=each['content'],
                metadata={"id": each['id']}
            ) for each in raw_text_list
        ]

        # construct raw text retrieval database
        construct_retrieval_database(
            dataset_name=dataset_name,
            database_id="raw-text",
            document_list=raw_text_document_list,
            mode="rebuild",
            encoder_model_name=encoder_model
        )

        # construct split text retrieval database
        task_id = os.getenv("TASK_ID") + "_" + dataset_name + "_" + generate_4_digit_uuid()
        split_docs_to_claims(db_path, raw_text_list, task_id, dataset_name == "legal-plus")
        split_result = read_split_result_from_db(db_path, task_id)
        construct_split_retrieval_database(dataset_name, split_result)
