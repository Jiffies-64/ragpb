import os

from main.dataset.dataset_chatdoctor_plus import ChatDoctorPlus
from main.dataset.dataset_legal_plus import LegalPlus
from main.utils.dataset_reader import CsvFileContentReader
from main.utils.constant import root_dir

DATASETS_NAME_TO_CLASS = {
    "chatdoctor-plus": ChatDoctorPlus,
    "legal-plus": LegalPlus
}

DATASETS_NAME_TO_DATA_LIST = {
    "chatdoctor-plus": CsvFileContentReader(
        file_path=os.path.join(root_dir, "dataset/chatdoctor-plus/chatdoctor-plus.csv"),
        id_column_name="name",
        content_column_name="text"
    ),
    "legal-plus": CsvFileContentReader(
        file_path=os.path.join(root_dir, "dataset/legal/legal_plus_300.csv"),
        id_column_name="id",
        content_column_name="complete_case"
    )
}
