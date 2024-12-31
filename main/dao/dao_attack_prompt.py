class AttackPrompt:
    def __init__(self, dataset_name, prompt_id, privacy_extraction_prompt, extract_response, patient_privacy_info, prompt,
                 attack_difficulty, response=""):
        self.dataset_name = dataset_name
        self.prompt_id = prompt_id
        self.attack_difficulty = attack_difficulty
        self.privacy_extraction_prompt = privacy_extraction_prompt
        self.extract_response = extract_response
        self.patient_privacy_info = patient_privacy_info  # "Historical legacy issues", where the expected privacy during the attack is stored
        self.prompt = prompt
        self.response = response

    def to_tuple(self):
        """
        Convert the object's attributes into a tuple, convenient for operations such as inserting into a database.
        """
        return (self.dataset_name, self.prompt_id, self.privacy_extraction_prompt, self.extract_response,
                self.patient_privacy_info, self.prompt, self.attack_difficulty, self.response)

    @staticmethod
    def insert_prompt(conn, prompt):
        """
        Insert the AttackPrompt object into the attack_prompt table.
        """
        cursor = conn.cursor()
        insert_query = "INSERT INTO attack_prompt (dataset_name, prompt_id, privacy_extraction_prompt, extract_response, patient_privacy_info, prompt, attack_difficulty, response) VALUES (?,?,?,?,?,?,?,?)"
        cursor.execute(insert_query, prompt.to_tuple())
        conn.commit()

    @staticmethod
    def retrieve_prompts(conn, condition=None):
        """
        Retrieve autoregressive-related records from the attack_prompt table, and can be filtered based on conditions.
        :param conn: Database connection
        :param condition: SQL WHERE clause condition (optional), e.g., "score > 50"
        """
        cursor = conn.cursor()
        if condition:
            select_query = f"SELECT * FROM attack_prompt WHERE {condition}"
        else:
            select_query = "SELECT * FROM attack_prompt"
        cursor.execute(select_query)
        results = cursor.fetchall()
        regressions = [AttackPrompt(*row) for row in results]

        return regressions

    @staticmethod
    def update_response_by_id(conn, dataset_name, prompt_id, new_response):
        """
        Update the corresponding response field value based on the given id.
        """
        cursor = conn.cursor()
        update_query = "UPDATE attack_prompt SET response =? WHERE dataset_name =? and prompt_id=?"
        cursor.execute(update_query, (new_response, dataset_name, prompt_id))
        conn.commit()