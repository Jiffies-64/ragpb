import sqlite3


class PersonalIdentification:
    def __init__(self, dataset_name, prompt_id, response, target, predict, score):
        self.dataset_name = dataset_name
        self.prompt_id = prompt_id
        self.response = response
        self.target = target
        self.predict = predict
        self.score = score

    def to_tuple(self):
        """
        Convert the object's attributes into a tuple, which is convenient for operations like inserting into a database.
        """
        return self.dataset_name, self.prompt_id, self.response, self.target, self.predict, self.score

    @staticmethod
    def insert_personal_identification(conn, personal_identification):
        """
        Insert the PersonalIdentification object into the personal_identification table.
        """
        cursor = conn.cursor()
        insert_query = "INSERT INTO personal_identification (dataset_name, prompt_id, response, target, predict, score) VALUES (?,?,?,?,?,?)"
        cursor.execute(insert_query, personal_identification.to_tuple())
        conn.commit()

    @staticmethod
    def retrieve_personal_identifications(conn, condition=None):
        """
        Retrieve personal identification related records from the personal_identification table, which can be filtered by condition.
        :param conn: Database connection
        :param condition: SQL WHERE clause condition (optional), for example "target = 'user1'"
        """
        cursor = conn.cursor()
        if condition:
            select_query = f"SELECT * FROM personal_identification WHERE {condition}"
        else:
            select_query = "SELECT * FROM personal_identification"
        cursor.execute(select_query)
        results = cursor.fetchall()
        identifications = [PersonalIdentification(*row) for row in results]

        return identifications