class TaskRecord:
    def __init__(self, id_, prompts_num, time, status):
        self.id = id_
        self.prompts_num = prompts_num
        self.time = time
        self.status = status

    def to_tuple(self):
        """
        Convert the object's attributes into a tuple for convenient operations like inserting into a database.
        """
        return self.id, self.prompts_num, self.time, self.status

    @staticmethod
    def insert_record(conn, record):
        """
        Insert the TaskRecord object into the task_record table.
        """
        cursor = conn.cursor()
        insert_query = "INSERT INTO task_record (id, prompts_num, time, status) VALUES (?,?,?,?)"
        cursor.execute(insert_query, record.to_tuple())
        conn.commit()