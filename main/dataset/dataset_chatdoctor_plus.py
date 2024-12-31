import copy
import json
from tqdm import tqdm

from main.dao.dao_attack_prompt import AttackPrompt
from main.dao.dao_task_record import TaskRecord
from main.dao.dao_lexical_overlap import LexicalOverlap
from main.dao.dao_personal_identification import PersonalIdentification
from main.dao.dao_self_regression import SelfRegression
from main.dataset.abstract_dataset import AbstractDataset
from main.evaluation.lexical_overlap.rouge_l import calculate_rouge_l
from main.evaluation.personal_identification.retrieval import calculate_personal_identification, calculate_percentage_in_top_k
from main.evaluation.self_regression.check_privacy_leaks import CheckResult, construct_privacy_check_prompt
from main.evaluation.self_regression.extracted_privacy_attr import PatientPrivacyInfo
from main.evaluation.self_regression.generate_attack_prompts import PrivacyAttackGenerator
from main.prompt.prompt_chatdoctor_plus import PromptChatDoctorPlus
from sentence_transformers import SentenceTransformer, util

from main.utils.retrieval_database import load_retrieval_database_from_parameter
from main.utils.uuid_util import generate_4_digit_uuid


class ChatDoctorPlus(AbstractDataset):
    prompt = PromptChatDoctorPlus

    def __init__(self, llm):
        super().__init__(llm, "chatdoctor-plus")
        # Load the retrieval database for raw text
        self.raw_rdb = load_retrieval_database_from_parameter(
            dataset_name="chatdoctor-plus",
            database_id="raw-text",
            encoder_model_name='all-MiniLM-L6-v2'
        )
        # Load the retrieval database for split text
        self.split_rdb = load_retrieval_database_from_parameter(
            dataset_name="chatdoctor-plus",
            database_id="split-text",
            encoder_model_name='all-MiniLM-L6-v2'
        )

    def prepare_prompt(self, dataset_name, row_id, statement):
        """
        Prepare the prompt for privacy extraction and attack.
        1. Build a privacy extraction prompt based on the statement.
        2. Get the LLM output for the extraction prompt.
        3. Generate attack prompts using the extracted patient information.
        4. Insert relevant information into the database.
        """
        # Build the privacy extraction prompt and extract information
        privacy_extraction_prompt = self.prompt.PRIVACY_EXTRACTION_TEMPLATE.format(medical_statement=statement)
        extract_response = self.llm.get_llm_output(privacy_extraction_prompt)
        patient_info = PatientPrivacyInfo.from_json_str(extract_response)

        # Build privacy attacks
        attack_generator = PrivacyAttackGenerator(pp=patient_info, pt=self.prompt.PRIVACY_ATTACH_TEMPLATES)
        attack_prompts = attack_generator.create_attack_prompts()

        # Store some parts of the prompt construction in the database
        for ap in attack_prompts:
            prompt_id = row_id + '_' + generate_4_digit_uuid()
            prompt = ap["prompt"]
            attack_difficulty = ap["attack_difficulty"]

            current_patient_info = copy.deepcopy(patient_info)

            # Remove sensitive information from the prompt if it exists
            if current_patient_info.name is not None and current_patient_info.name in prompt:
                current_patient_info.name = None
            if current_patient_info.gender is not None and current_patient_info.gender in prompt:
                current_patient_info.gender = None
            if current_patient_info.age and str(current_patient_info.age) in prompt:
                current_patient_info.age = None
            symptoms = current_patient_info.symptoms
            if symptoms:
                for symptom in current_patient_info.symptoms:
                    if str(symptom.symptom_name) in prompt:
                        symptoms.remove(symptom)
            current_patient_info.symptoms = symptoms

            record = AttackPrompt(
                dataset_name=dataset_name,
                prompt_id=prompt_id,
                privacy_extraction_prompt=privacy_extraction_prompt,
                extract_response=extract_response,
                patient_privacy_info=json.dumps(current_patient_info.to_dict()),
                prompt=prompt,
                attack_difficulty=attack_difficulty
            )
            AttackPrompt.insert_prompt(self.conn, record)

    def calculate_lexical_overlap(self, dataset_name, attack_prompts):
        """
        Calculate the lexical overlap using ROUGE-L for each attack prompt.
        1. Skip if the response is empty.
        2. Find similar documents using the raw retrieval database.
        3. Calculate ROUGE-L scores.
        4. Insert the lexical overlap object into the database.
        """
        for ap in attack_prompts:
            if not ap.response:
                continue
            similar_docs = self.raw_rdb.similarity_search(ap.response, k=3)
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
        Calculate the semantic similarity between two texts (question and answer) based on cosine similarity.
        :param question: User's question (string)
        :param answer: Model-generated answer (string)
        :return: Semantic similarity score (0~1)
        """
        question = answer = ""
        # Use a lightweight sentence transformer model for encoding
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Encode the input texts into vectors
        embeddings = sentence_model.encode([question, answer], convert_to_tensor=True)
        # Calculate the cosine similarity
        similarity_score = util.cos_sim(embeddings[0], embeddings[1])
        return similarity_score.item()

    def calculate_personal_identification(self, dataset_name, attack_prompts):
        """
        Calculate personal identification scores.
        1. Skip if the response is empty.
        2. Calculate personal identification using the split retrieval database.
        3. Calculate the score based on the top-k results.
        4. Insert the personal identification object into the database.
        """
        for ap in tqdm(attack_prompts):
            if not ap.response:
                continue
            topk_predict = calculate_personal_identification(self.split_rdb, ap.response)
            key = ap.prompt_id.split('_')[0]
            score = calculate_percentage_in_top_k(topk_predict, key)

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
        Calculate self-regression results.
        1. Skip if the response is empty.
        2. Construct a privacy check prompt.
        3. Get the LLM output for the privacy check prompt.
        4. Create a SelfRegression object and insert it into the database.
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