import sqlite3

from main.dao.dao_lexical_overlap import LexicalOverlap
from main.dao.dao_personal_identification import PersonalIdentification
from main.dao.dao_self_regression import SelfRegression

NAME_DAO_MAPPER = {
    "lexical_overlap": LexicalOverlap,
    "personal_identification": PersonalIdentification,
    "self_regression": SelfRegression,
}


def init_temp_database_for_claims(db_path):
    """
    Initialize the temporary database for claims.
    This function creates two tables: 'claims' and 'processing_status' if they do not already exist.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create the 'claims' table with columns 'task_id', 'id', and 'claim' if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS claims (
            task_id TEXT,
            id TEXT,
            claim TEXT
        )
    ''')
    # Create the 'processing_status' table with columns 'id' and 'processed_index' if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_status (
            id TEXT PRIMARY KEY,
            processed_index INTEGER
        )
    ''')
    conn.commit()


def initialize_database(db_path):
    """
    Initialize the main database.
    This function creates several tables related to different metrics and data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create the 'task_record' table with columns 'id', 'prompts_num', 'time', and 'status' if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_record (
            id TEXT,
            prompts_num INTEGER,
            time TEXT,
            status TEXT
        )
    ''')
    # Create the 'attack_prompt' table with columns related to attack prompts and their details if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS attack_prompt (
            dataset_name TEXT,
            prompt_id TEXT,
            privacy_extraction_prompt TEXT,
            extract_response TEXT, 
            patient_privacy_info TEXT,
            prompt TEXT,
            attack_difficulty INT,
            response TEXT
        )
    ''')
    # Create the 'lexical_overlap' table with columns related to lexical overlap metrics if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS lexical_overlap (
            dataset_name TEXT,
            prompt_id TEXT,
            response TEXT,
            precision FLOAT,
            recall FLOAT,
            fmeasure FLOAT
        )
    ''')
    # Create the 'self_regression' table with columns related to self-regression data if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS self_regression (
            dataset_name TEXT,
            prompt_id TEXT,
            response TEXT,
            privacy_check_prompt TEXT,
            privacy_check_response TEXT,
            score INT,
            check_analysis TEXT
        )
    ''')
    # Create the 'semantic_similarity' table with columns related to semantic similarity if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS semantic_similarity (
            id TEXT,
            prompt TEXT,
            response TEXT,
            score INT
        )
    ''')
    # Create the 'personal_identification' table with columns related to personal identification if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS personal_identification (
            dataset_name TEXT,
            prompt_id TEXT,
            response TEXT,
            target TEXT,
            predict TEXT,
            score INT
        )
    ''')
    # Create the 'task_utility' table with columns related to task utility and its score if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS task_utility (
            id TEXT,
            prompt TEXT,
            response TEXT,
            score INT,
            reason TEXT
        )
    ''')
    # Create the 'text_coherence' table with columns related to text coherence if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS text_coherence (
            id TEXT,
            prompt TEXT,
            response TEXT,
            score INT
        )
    ''')
    # Create the 'construct_loss' table with columns related to construct loss if it does not exist
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS construct_loss (
            id TEXT,
            prompt TEXT,
            response TEXT,
            score INT
        )
    ''')
    conn.commit()