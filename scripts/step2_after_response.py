import os
import argparse
from dotenv import load_dotenv

from main.dataset.__all__ import DATASETS_NAME_TO_CLASS
from main.utils.llm_factory import LLMManager
from main.utils.constant import root_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="You can choose from the following datasets: chatdoctor-plus, legal-plus. "
             "This parameter specifies the dataset(s) to be used in the process.",
        required=False,
        default=["legal-plus", ],  # "chatdoctor-plus", "legal-plus"
        type=list
    )
    parser.add_argument(
        "--test-points",
        help="You can choose from: lexical_overlap, semantic_similarity, personal_identification, self_regression,"
             "task_utility, text_coherence, construct_loss",
        required=False,
        default=["lexical_overlap", "self_regression", "personal_identification"],
        type=list,
    )
    return parser.parse_args()


if __name__ == '__main__':
    # load environments and parse arguments
    load_dotenv(os.path.join(root_dir, '.env'))
    task_id = os.getenv("TASK_ID")
    args = parse_args()

    # get llm
    llm = LLMManager(LLMManager.ModelEnum.GPT4o)

    for dataset_name in args.dataset:
        paths = [
            "output/{}",
            f"store/{dataset_name}/database/{{}}.sqlite"
        ]
        for path in paths:
            if not os.path.exists(os.path.join(root_dir, path.format(task_id))):
                raise Exception(
                    f"""
                    TASK_ID (current value is `{task_id}`) specified in `.env` does not exist! 
                    1. Please copy and rename `store/{dataset_name}/database/default.sqlite`
                    2. Please copy and rename `output/default`
                    3. Please rename TASK_ID in `.env`
                    """
                )

        ds = DATASETS_NAME_TO_CLASS[dataset_name](llm)
        ds.store_response(f"response_template_{task_id}.csv", dataset_name=dataset_name)
        ds.stage_2_calculate(args.test_points, dataset_name)

        ds.generate_report(args.test_points, dataset_name)
