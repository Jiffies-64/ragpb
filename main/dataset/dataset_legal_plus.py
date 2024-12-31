import json
import random
from tqdm import tqdm

from main.dao.dao_attack_prompt import AttackPrompt
from main.dao.dao_lexical_overlap import LexicalOverlap
from main.dao.dao_personal_identification import PersonalIdentification
from main.dao.dao_self_regression import SelfRegression
from main.dataset.abstract_dataset import AbstractDataset
from main.evaluation.lexical_overlap.rouge_l import calculate_rouge_l
from main.evaluation.personal_identification.retrieval import calculate_personal_identification, calculate_percentage_in_top_k
from main.evaluation.self_regression.check_privacy_leaks import CheckResult, construct_privacy_check_prompt
from main.prompt.prompt_legal_plus import PromptLegalPlus
from main.utils.retrieval_database import load_retrieval_database_from_parameter
from main.utils.uuid_util import generate_4_digit_uuid


class LegalPlus(AbstractDataset):

    def __init__(self, llm):
        super().__init__(llm, "legal-plus")
        # Load the retrieval database for raw text
        self.raw_rdb = load_retrieval_database_from_parameter(
            dataset_name="legal-plus",
            database_id="raw-text",
            encoder_model_name='all-MiniLM-L6-v2'
        )
        # Load the retrieval database for split text
        self.split_rdb = load_retrieval_database_from_parameter(
            dataset_name="legal-plus",
            database_id="split-text",
            encoder_model_name='all-MiniLM-L6-v2'
        )

    def prepare_prompt(self, dataset_name, row_id, statement):
        """
        Prepare the prompt for the legal-plus dataset.
        statement is a dictionary containing original content, privacy contents, and complete case.
        It selects some privacy contents randomly and constructs an attack prompt.
        The selected privacy contents and the remaining privacy contents are stored.
        An attack prompt is generated using the LLM and stored in the AttackPrompt table.
        """
        # Extract original content and privacy contents from the statement
        original_content = statement["original_content"]
        privacy_contents = eval(statement["privacy_contents"])

        # Randomly select a subset of privacy contents based on the number of available contents
        if len(privacy_contents) > 5:
            selected_privacy_contents = random.sample(privacy_contents, 3)
        elif 3 < len(privacy_contents) <= 5:
            selected_privacy_contents = random.sample(privacy_contents, 2)
        elif 1 < len(privacy_contents) <= 3:
            selected_privacy_contents = random.sample(privacy_contents, 1)
        else:
            return

        # Identify the privacy contents that are not selected
        target_privacy_list = [each for each in privacy_contents if each not in selected_privacy_contents]

        # Format the prompt construction template with selected privacy contents
        selected_privacy_content_str = " ".join(selected_privacy_contents)
        construction_prompt = PromptLegalPlus.PROMPT_CONSTRUCTION_TEMPLATE.format(
            raw_content=original_content,
            selected_privacy=selected_privacy_content_str
        )

        # Get the LLM output for the construction prompt
        attack_prompt = self.llm.get_llm_output(construction_prompt)
        prompt_id = row_id + "<SEP>" + generate_4_digit_uuid()
        record = AttackPrompt(
            dataset_name=dataset_name,
            prompt_id=prompt_id,
            privacy_extraction_prompt=None,
            extract_response=str(selected_privacy_contents),
            patient_privacy_info=str(target_privacy_list),
            prompt=attack_prompt,
            attack_difficulty=-1
        )
        AttackPrompt.insert_prompt(self.conn, record)

    def calculate_lexical_overlap(self, dataset_name, attack_prompts):
        """
        Calculate the lexical overlap for each attack prompt.
        For each prompt, if the response is not empty, it searches for similar documents using the raw retrieval database.
        It calculates ROUGE-L scores (precision, recall, fmeasure) and stores the result in the LexicalOverlap table.
        """
        for ap in attack_prompts:
            if not ap.response:
                continue
            similar_docs = self.raw_rdb.similarity_search(ap.response, k=10)
            retrieval_text = "; ".join(doc.page_content for doc in similar_docs)
            rouge_l = calculate_rouge_l(ap.response, retrieval_text)
            lexical_overlap = LexicalOverlap(
                dataset_name=dataset_name,
                prompt_id=ap.prompt_id,
                response=ap.response,
                precision=rouge_l.precision,
                recall=rouge_l.recall,
                fmeasure=rouge_l.fmeasure
            )
            LexicalOverlap.insert_lexical_overlap(self.conn, lexical_overlap)

    def calculate_semantic_similarity(self, dataset_name):
        """
        This method is currently empty. It might be intended to calculate semantic similarity in the future.
        """
        pass

    def calculate_personal_identification(self, dataset_name, attack_prompts):
        """
        Calculate personal identification scores for each attack prompt.
        For each prompt, if the response is not empty, it calculates personal identification using the split retrieval database.
        It calculates a score based on top-k predictions and stores the result in the PersonalIdentification table.
        """
        for ap in tqdm(attack_prompts):
            if not ap.response:
                continue
            topk_predict = calculate_personal_identification(self.split_rdb, ap.response, 5)
            key = ap.prompt_id.split("<SEP>")[0]
            score = calculate_percentage_in_top_k(topk_predict, key, 3)

            personal_identification_obj = PersonalIdentification(
                dataset_name=dataset_name,
                prompt_id=ap.prompt_id,
                response=ap.response,
                target=ap.prompt_id.split('-')[0],
                predict=json.dumps(topk_predict),
                score=score
            )
            PersonalIdentification.insert_personal_identification(self.conn, personal_identification_obj)

    def calculate_self_regression(self, dataset_name, attack_prompts):
        """
        Calculate self-regression for each attack prompt.
        For each prompt, if the response is not empty, it constructs a privacy check prompt.
        It gets the LLM output for the privacy check prompt, parses the result, and stores it in the SelfRegression table.
        """
        for ap in tqdm(attack_prompts):
            if not ap.response:
                continue

            unused_privacy = ap.patient_privacy_info
            privacy_check_prompt = construct_privacy_check_prompt(
                ap.response,
                unused_privacy
            )
            privacy_check_response = self.llm.get_llm_output(privacy_check_prompt)
            check_result = CheckResult.from_json_str(privacy_check_response)
            self_regression_obj = SelfRegression(
                dataset_name=dataset_name,
                prompt_id=ap.prompt_id,
                response=ap.response,
                privacy_check_prompt=privacy_check_prompt,
                privacy_check_response=privacy_check_response,
                score=check_result.score,
                check_analysis=check_result.analysis
            )
            SelfRegression.insert_self_regression(self.conn, self_regression_obj)

    def get_row_content(self, row):
        """
        Extract relevant information from a row and return it as a dictionary.
        """
        return {
            'original_content': row['original_content'],
            'privacy_contents': row['privacy_contents'],
            'complete_case': row['content']
        }