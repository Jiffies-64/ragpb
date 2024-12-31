import sqlite3


class LexicalOverlap:
    def __init__(
            self,
            dataset_name,
            prompt_id,
            response,
            precision,
            recall,
            fmeasure
    ):
        self.dataset_name = dataset_name
        self.prompt_id = prompt_id
        self.response = response
        self.precision = precision
        self.recall = recall
        self.fmeasure = fmeasure

    def to_tuple(self):
        """
        Convert the object's attributes into a tuple, which is convenient for operations such as inserting into a database.
        """
        return self.dataset_name, self.prompt_id, self.response, self.precision, self.recall, self.fmeasure

    @staticmethod
    def insert_lexical_overlap(conn, lexical_overlap):
        """
        Insert the LexicalOverlap object into the lexical_overlap table.
        """
        cursor = conn.cursor()
        insert_query = "INSERT INTO lexical_overlap (dataset_name, prompt_id, response, precision, recall, fmeasure) VALUES (?,?,?,?,?,?)"
        cursor.execute(insert_query, lexical_overlap.to_tuple())
        conn.commit()

    @staticmethod
    def retrieve_lexical_overlaps(conn, condition=None):
        """
        Retrieve lexical overlap related records from the lexical_overlap table, and it can be filtered according to conditions.
        :param conn: Database connection
        :param condition: SQL WHERE clause condition (optional), e.g., "prompt LIKE '%test%'"
        """
        cursor = conn.cursor()
        if condition:
            select_query = f"SELECT * FROM lexical_overlap WHERE {condition}"
        else:
            select_query = "SELECT * FROM lexical_overlap"
        cursor.execute(select_query)
        results = cursor.fetchall()
        overlaps = [LexicalOverlap(*row) for row in results]
        return overlaps

    @staticmethod
    def update_response_by_id(conn, id_, new_response):
        """
        Update the corresponding response field value based on the given id.
        """
        cursor = conn.cursor()
        update_query = "UPDATE lexical_overlap SET response =? WHERE id =?"
        cursor.execute(update_query, (new_response, id_))
        conn.commit()
