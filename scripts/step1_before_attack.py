import os.path
import argparse
from dotenv import load_dotenv

from main.dataset.__all__ import DATASETS_NAME_TO_CLASS, DATASETS_NAME_TO_DATA_LIST
from main.utils.llm_factory import LLMManager
from main.utils.constant import root_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="You can choose from the following datasets: chatdoctor-plus, legal-plus. "
             "This parameter specifies the dataset(s) to be used in the process.",
        required=False,
        default=["legal-plus",],  # "chatdoctor-plus",
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
    # load environments and parse arguments
    load_dotenv(os.path.join(root_dir, '.env'))

    args = parse_args()
    
    # get llm
    llm = LLMManager(LLMManager.ModelEnum.GPT4o)
    
    for dataset_name in args.dataset:
    
        ds = DATASETS_NAME_TO_CLASS[dataset_name](llm)
        data_list = DATASETS_NAME_TO_DATA_LIST[dataset_name].read_content(args.limit)
        ds.stage_1_prepare(data_list, dataset_name)
    
        ds.export_prompt_to_csv(dataset_name)
