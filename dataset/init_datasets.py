import argparse
import os
import shutil

import kagglehub
from datasets import load_dataset

datasets_to_download = {
    "tboyle10/medicaltranscriptions": "medical-transcriptions",
    "alejopaullier/pii-external-dataset": "pii-external-dataset",
    # "Medilora/mimic_iii_diagnosis_anonymous": "mimic_iii_diagnosis_anonymous",
}


def download_kaggle_dataset(dataset_name, local_path):
    """
    从Kaggle下载数据集到指定的本地路径
    """
    try:
        os.makedirs(local_path, exist_ok=True)
        os.chdir(local_path)
        path = kagglehub.dataset_download(dataset_name)
        print(f"Path to {dataset_name} dataset files:", path)
    except Exception as e:
        print(f"Error downloading {dataset_name} dataset: {str(e)}")


def download_huggingface_dataset(dataset_name, local_path):
    """
    从Hugging Face下载数据集到指定的本地路径
    """
    try:
        os.makedirs(local_path, exist_ok=True)
        os.chdir(local_path)
        dataset = load_dataset(dataset_name)
        print(f"Successfully downloaded {dataset_name} dataset.")
        print(dataset)
    except Exception as e:
        print(f"Error downloading {dataset_name} dataset: {str(e)}")


def main(args):
    if not os.path.exists(os.path.expanduser('~/.kaggle')):
        os.makedirs(os.path.expanduser('~/.kaggle'))

    root_kaggle_file = os.path.join(os.path.dirname(__file__), '..', 'kaggle.json')
    if os.path.exists(root_kaggle_file):
        shutil.copy(root_kaggle_file, os.path.expanduser('~/.kaggle/kaggle.json'))
    else:
        print("请将 kaggle.json 文件放置在项目根目录下", root_kaggle_file)

    # 确保文件权限正确
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

    if args.dataset == "all":
        for dataset_name, local_folder_name in datasets_to_download.items():
            if "kaggle.com" in dataset_name:
                download_kaggle_dataset(dataset_name, local_folder_name)
            else:
                download_huggingface_dataset(dataset_name, local_folder_name)
    else:
        if args.dataset in datasets_to_download:
            local_folder_name = datasets_to_download[args.dataset]
            if "kaggle.com" in args.dataset:
                download_kaggle_dataset(args.dataset, local_folder_name)
            else:
                download_huggingface_dataset(args.dataset, local_folder_name)
        else:
            print(f"Dataset {args.dataset} not recognized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets based on specified names.")
    parser.add_argument("--dataset", default="all", type=str,
                        help="Name of the dataset to download, use 'all' to download all datasets")
    args = parser.parse_args()
    main(args)
