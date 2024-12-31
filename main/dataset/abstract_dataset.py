import csv
import os
import sqlite3
import time

import pandas as pd

from main.dao.dao_attack_prompt import AttackPrompt
from main.dao.dao_lexical_overlap import LexicalOverlap
from main.dao.dao_personal_identification import PersonalIdentification
from main.dao.dao_self_regression import SelfRegression
from main.dao.dao_task_record import TaskRecord
from main.dao.database import initialize_database
from main.prompt.abstract_prompt import AbstractPrompt
from tqdm import tqdm

from main.utils.constant import root_dir


class AbstractDataset:
    dataset: str = None
    prompt: AbstractPrompt = None

    def __init__(self, llm, dataset):
        """
        Initialize the AbstractDataset object.
        :param llm: The language model.
        :param dataset: The name of the dataset.
        """
        self.llm = llm
        self.dataset = dataset
        # Construct the directory path for the database
        db_dir = os.path.join(root_dir, 'store', self.dataset, 'database')
        # Create the directory if it does not exist
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        # Create the full path for the SQLite database file
        db_path = os.path.join(db_dir, os.getenv("TASK_ID") + '.sqlite')
        # Initialize the database
        initialize_database(db_path)
        # Connect to the database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def stage_1_prepare(self, raw_data_list, dataset_name):
        """
        Prepare data in stage 1.
        :param raw_data_list: The list of raw data.
        :param dataset_name: The name of the dataset.
        """
        total_lines = len(raw_data_list)
        # Iterate through the raw data list with a progress bar
        for each_line in tqdm(raw_data_list, total=total_lines, desc="Stage 1 Prepare"):
            row_id = each_line['id']
            row_content = self.get_row_content(each_line)
            self.prepare_prompt(dataset_name, row_id, row_content)

    def stage_2_calculate(self, test_points, dataset_name):
        """
        Perform calculations in stage 2.
        :param test_points: A list of test points.
        :param dataset_name: The name of the dataset.
        """
        # Retrieve attack prompts from the database
        attack_prompts = AttackPrompt.retrieve_prompts(self.conn, f"dataset_name = '{dataset_name}'")
        # Perform calculations based on the test points
        if 'lexical_overlap' in test_points:
            self.calculate_lexical_overlap(dataset_name, attack_prompts)
        if 'self_regression' in test_points:
            self.calculate_self_regression(dataset_name, attack_prompts)
        if 'personal_identification' in test_points:
            self.calculate_personal_identification(dataset_name, attack_prompts)

    def export_prompt_to_csv(self, dataset_name):
        """
        Export prompts to a CSV file.
        :param dataset_name: The name of the dataset.
        """
        task_id = os.getenv("TASK_ID")
        # Define paths for the prompt and response template CSV files
        prompt_output_path = os.path.join(
            root_dir, 'output', task_id, f'prompt_{dataset_name}.csv'
        )
        response_template_path = os.path.join(
            root_dir, 'output', task_id, f'response_template_{dataset_name}.csv'
        )
        prompt_header = ['id', 'prompt']
        response_header = ['id', 'response']
        rows = []
        # Retrieve prompts from the database
        prompts = AttackPrompt.retrieve_prompts(self.conn, f"dataset_name = '{dataset_name}'")
        # Insert a task record
        prompts_num = len(prompts)
        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        tr = TaskRecord(dataset_name, prompts_num, local_time, 1)
        TaskRecord.insert_record(self.conn, tr)
        # Prepare rows for the CSV file
        for p in prompts:
            rows.append([p.prompt_id, p.prompt])
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(prompt_output_path), exist_ok=True)
        # Write the prompt CSV file
        with open(prompt_output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(prompt_header)
            writer.writerows(rows)
        # Write the response template CSV file
        with open(response_template_path, 'w', newline='', encoding='utf-8-sig') as template:
            writer = csv.writer(template)
            writer.writerow(response_header)
            writer.writerows([r[0], None] for r in rows)

    def store_response(self, response_csv_file, dataset_name=None, eval_col_name="eval", id_col_name="id",
                      resp_col_name="response"):
        """
        Store responses from a CSV file.
        :param response_csv_file: The path to the response CSV file.
        :param dataset_name: The name of the dataset.
        :param eval_col_name: The name of the evaluation column.
        :param id_col_name: The name of the ID column.
        :param resp_col_name: The name of the response column.
        """
        response_path = os.path.join(root_dir, "user_upload", response_csv_file)
        df = pd.read_csv(response_path)
        # Update the response for each row in the DataFrame
        for index, row in df.iterrows():
            prompt_id = row[id_col_name]
            response = row[resp_col_name]
            AttackPrompt.update_response_by_id(self.conn, dataset_name, prompt_id, response)

    def generate_report(self, test_points, dataset_name):
        """
        Generate a report in CSV format.
        :param test_points: A list of test points.
        :param dataset_name: The name of the dataset.
        """
        report_path = os.path.join(root_dir, 'output', os.getenv("TASK_ID"), f'report_{dataset_name}.csv')
        report_header = ['eval', 'average']
        rows = []
        test_point_info = {
            'lexical_overlap': {
                'retrieve_func': LexicalOverlap.retrieve_lexical_overlaps,
                'fields': ['precision', 'recall', 'fmeasure']
            },
            'self_regression': {
                'retrieve_func': SelfRegression.retrieve_self_regressions,
                'fields': ['score']
            },
            'personal_identification': {
                'retrieve_func': PersonalIdentification.retrieve_personal_identifications,
                'fields': ['score']
            }
        }
        # Generate report rows based on the test points
        for test_point in test_points:
            if test_point in test_point_info:
                retrieved_data = test_point_info[test_point]['retrieve_func'](self.conn)
                for field in test_point_info[test_point]['fields']:
                    values = [getattr(data, field) for data in retrieved_data if getattr(data, field)!= -1]
                    average = self._calculate_average(values)
                    rows.append([f"{test_point}_average_{field}", average])
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        # Write the report CSV file
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(report_header)
            writer.writerows(rows)

    def _calculate_average(self, values, default_value=-1):
        """
        Calculate the average of a list of values.
        :param values: The list of values.
        :param default_value: The default value to return if the list is empty.
        :return: The average value.
        """
        count = sum(1 for v in values if v!= default_value)
        if count > 0:
            return sum(v for v in values if v!= default_value) / count
        return default_value

    def prepare_prompt(self, dataset_name, row_id, statement):
        """
        Prepare a prompt. This method is intended to be overridden by subclasses.
        :param dataset_name: The name of the dataset.
        :param row_id: The ID of the row.
        :param statement: The statement.
        """
        pass

    def calculate_lexical_overlap(self, dataset_name, attack_prompts):
        """
        Calculate lexical overlap. This method is intended to be overridden by subclasses.
        :param dataset_name: The name of the dataset.
        :param attack_prompts: A list of attack prompts.
        """
        pass

    def calculate_semantic_similarity(self, dataset_name):
        """
        Calculate semantic similarity. This method is intended to be overridden by subclasses.
        """
        pass

    def calculate_personal_identification(self, dataset_name, attack_prompts):
        """
        Calculate personal identification. This method is intended to be overridden by subclasses.
        :param dataset_name: The name of the dataset.
        :param attack_prompts: A list of attack prompts.
        """
        pass

    def calculate_self_regression(self, dataset_name, attack_prompts):
        """
        Calculate self-regression. This method is intended to be overridden by subclasses.
        :param dataset_name: The name of the dataset.
        :param attack_prompts: A list of attack prompts.
        """
        pass

    def get_row_content(self, row):
        """
        Get the content of a row.
        :param row: The row data.
        :return: The content of the row.
        """
        return row['content']