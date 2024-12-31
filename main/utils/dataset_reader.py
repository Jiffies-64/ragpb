import os
import uuid
import json
import pandas as pd


class FileContentReader:
    """
    Base class for handling the common logic of reading files or directories.
    """

    def __init__(self, file_path):
        """
        Initialization method. Checks the type of the incoming file path parameter and confirms whether the path exists.

        Args:
            file_path (str): The absolute path of a file or a directory.
        """
        if not isinstance(file_path, str):
            raise TypeError("The parameter file_path must be of string type.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified path {file_path} does not exist.")
        self.file_path = file_path

    def read_content(self, limit):
        """
        A unified method provided externally for reading content. The specific logic will be implemented by subclasses.

        Returns:
            list: A list containing formatted data. Each element is a dictionary that includes 'id' and 'content' and other related key-value pairs.
        """
        raise NotImplementedError("Subclasses need to implement this method.")


class TxtFileContentReader(FileContentReader):
    """
    A class used to handle the reading and formatting of.txt files, inheriting from the FileContentReader base class. It can specify the line break.
    """

    def __init__(self, file_path, line_break='\n'):
        """
        Initialization method. Calls the initialization method of the parent class to check the file path and initializes the line break used to split the content.

        Args:
            file_path (str): The absolute path of a file or a directory.
            line_break (str): The line break used to split the content of.txt files. The default is '\n', and it can also be specified as '\n\n', etc.
        """
        super().__init__(file_path)
        self.line_break = line_break

    def read_content(self, limit=-1):
        """
        Reads the content of a.txt file and returns a formatted list according to the rules.

        Returns:
            list: A list containing formatted data. Each element is a dictionary that includes 'id' and 'content' and other related key-value pairs.
        """
        result = []
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if self.line_break in content:
                parts = content.split(self.line_break)
                for part in parts:
                    if limit != -1 and count >= limit:
                        break
                    data_id = str(uuid.uuid4())
                    result.append({"id": data_id, "content": part.strip()})
                    count += 1
            else:
                lines = content.splitlines()
                for line in lines:
                    if limit != -1 and count >= limit:
                        break
                    result.append({"id": str(uuid.uuid4()), "content": line})
                    count += 1
        return result


class JsonlFileContentReader(FileContentReader):
    """
    A class used to handle the reading and formatting of.jsonl files, inheriting from the FileContentReader base class. It can specify the id column name.
    """

    def __init__(self, file_path, id_column_name='id', content_column_name='content'):
        """
        Initialization method. Calls the initialization method of the parent class to check the file path and initializes the column name used to obtain the id.

        Args:
            file_path (str): The absolute path of a file or a directory.
            id_column_name (str): The column name used as the id in the JSONL file. The default is 'id', and it can be customized.
        """
        super().__init__(file_path)
        self.id_column_name = id_column_name
        self.content_column_name = content_column_name

    def read_content(self, limit=-1):
        """
        Reads the content of a.jsonl file and returns a formatted list according to the rules.

        Returns:
        list: A list containing formatted data. Each element is a dictionary that includes 'id' and other key-value pairs parsed from the JSON data.
        """
        result = []
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if limit != -1 and count >= limit:
                    break
                try:
                    json_data = json.loads(line)
                    data_id = json_data.get(self.id_column_name, str(uuid.uuid4()))
                    data_content = json_data.get(self.content_column_name)
                    if data_content is None:
                        raise KeyError(f"Column '{self.content_column_name}' not found in the JSON data.")
                    # Extract other key-value pairs from the JSON data, excluding the already processed 'id' key (if any).
                    other_info_dict = {k: v for k, v in json_data.items() if
                                       k != self.id_column_name and k != self.content_column_name}
                    result.append({"id": data_id, "content": data_content, **other_info_dict})
                    count += 1
                except json.JSONDecodeError:
                    continue
        return result


class CsvFileContentReader(FileContentReader):
    """
    A class used to handle the reading and formatting of.csv files, inheriting from the FileContentReader base class. It can specify the id column name and content column name.
    """

    def __init__(self, file_path, id_column_name='id', content_column_name='content'):
        """
        Initialization method. Calls the initialization method of the parent class to check the file path and initializes the column names used to obtain the id and content.

        Args:
        file_path (str): The absolute path of a file or a directory.
        id_column_name (str): The column name used as the id in the CSV file. The default is 'id', and it can be customized.
        content_column_name (str): The column name used to obtain the content in the CSV file. The default is 'content', and it can be customized.
        """
        super().__init__(file_path)
        self.id_column_name = id_column_name
        self.content_column_name = content_column_name

    def read_content(self, limit=-1):
        """
        Reads the content of a.csv file and returns a formatted list according to the rules.

        Returns:
            list: A list containing formatted data. Each element is a dictionary that includes 'id' and 'content' and other related key-value pairs.
        """
        result = []
        count = 0
        df = pd.read_csv(self.file_path)
        columns = df.columns.tolist()  # Get the list of column names of the CSV file.
        id_column_exists = self.id_column_name in columns  # Check whether the specified id column name exists.
        content_column_exists = self.content_column_name in columns  # Check whether the specified content column name exists.

        for index, row in df.iterrows():
            if limit != -1 and count >= limit:
                break
            if id_column_exists:
                data_id = row[
                    self.id_column_name]  # If the column with the specified id column name exists, use its value as the id.
            else:
                data_id = str(uuid.uuid4())

            data_content = row.get(self.content_column_name)
            if data_content is None and content_column_exists:
                raise KeyError(f"Column '{self.content_column_name}' not found in the CSV data at row {index}.")

            other_info_dict = {k: v for k, v in row.items() if
                               k != self.id_column_name and k != self.content_column_name}
            result.append({"id": data_id, "content": data_content, **other_info_dict})
            count += 1

        return result


class DirectionContentReader(FileContentReader):
    def read_content(self, limit=-1):
        """
        Traverses all.txt files under the directory, reads their contents, and returns a formatted list according to the rules.

        Returns:
            list: A list containing formatted data. Each element is a dictionary that includes 'id' and 'content' and other related key-value pairs.
        """
        result = []
        count = 0
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                file_full_path = os.path.join(root, file)
                if os.path.splitext(file_full_path)[-1].lower() == '.txt':
                    if limit != -1 and count >= limit:
                        break
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        result.append({"id": file, "content": content})
                    count += 1
        return result
