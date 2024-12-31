import sqlite3


class SelfRegression:
    def __init__(
            self,
            dataset_name,
            prompt_id,
            response="",
            privacy_check_prompt="",
            privacy_check_response="",
            score=-1,
            check_analysis=""
    ):
        self.dataset_name = dataset_name
        self.prompt_id = prompt_id
        self.response = response
        self.privacy_check_prompt = privacy_check_prompt
        self.privacy_check_response = privacy_check_response
        self.score = score
        self.check_analysis = check_analysis

    def to_tuple(self):
        """
        Convert the object's attributes into a tuple for operations such as database insertion.
        """
        return (self.dataset_name, self.prompt_id, self.response, self.privacy_check_prompt, self.privacy_check_response,
                self.score, self.check_analysis)

    @staticmethod
    def insert_self_regression(conn, self_regression):
        """
        Insert the SelfRegression object into the self_regression table.
        """
        cursor = conn.cursor()
        insert_query = "INSERT INTO self_regression (dataset_name, prompt_id, response, privacy_check_prompt, privacy_check_response, score, check_analysis) VALUES (?,?,?,?,?,?,?)"
        cursor.execute(insert_query, self_regression.to_tuple())
        conn.commit()

    @staticmethod
    def retrieve_self_regressions(conn, condition=None):
        """
        Retrieve self-regression related records from the self_regression table, which can be filtered based on conditions.
        :param conn: Database connection
        :param condition: SQL WHERE clause condition (optional), e.g., "score > 50"
        """
        cursor = conn.cursor()
        if condition:
            select_query = f"SELECT * FROM self_regression WHERE {condition}"
        else:
            select_query = "SELECT * FROM self_regression"
        cursor.execute(select_query)
        results = cursor.fetchall()
        regressions = [SelfRegression(*row) for row in results]

        return regressions

    @staticmethod
    def update_response_by_id(conn, id_, new_response):
        """
        Update the corresponding response field value based on the given id.
        """
        cursor = conn.cursor()
        update_query = "UPDATE self_regression SET response =? WHERE id =?"
        cursor.execute(update_query, (new_response, id_))
        conn.commit()