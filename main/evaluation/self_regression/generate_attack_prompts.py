import random
from typing import List, Optional, Dict

from main.evaluation.self_regression.extracted_privacy_attr import PatientPrivacyInfo


class PrivacyAttackGenerator:
    def __init__(self, pp: PatientPrivacyInfo, pt: dict):
        self.patient_privacy_info = pp
        self.GENERAL_DIRECT_MATCH_TEMPLATE = pt["GENERAL_DIRECT_MATCH"]
        self.PROBABILITY_QUERY_TEMPLATE = pt["PROBABILITY_QUERY"]

    def create_attack_prompts(self) -> List[Dict[str, Optional[str]]]:
        """
        Create various attack prompts by selecting combinations of symptoms with different levels of difficulty.
        """
        prompts = []

        privacy_dict = {
            key: value
            for key, value in self.patient_privacy_info.additional_privacy_attributes.items()
            if value is not None
        }
        for symptom in self.patient_privacy_info.symptoms:
            privacy_dict[symptom.symptom_name] = symptom.symptom_details
        privacy_count = len(privacy_dict)
        if privacy_count > 2:
            # 随机选择一组症状或其他隐私的组合
            combo = random.sample(list(privacy_dict.items()), min(privacy_count, 4))

            symptoms_descriptions = '; '.join(
                [f"{k} with details: {v}" for k, v in combo]
            )

            prompts.append({
                "prompt": random.choice(self.PROBABILITY_QUERY_TEMPLATE).format(symptoms=symptoms_descriptions),
                "attack_difficulty": 2
            })

        if self.patient_privacy_info.name:
            pii_descriptions = "name: {}, age:{}, gender: {}".format(
                self.patient_privacy_info.name,
                self.patient_privacy_info.age,
                self.patient_privacy_info.gender
            )
            prompts.append({
                "prompt": random.choice(self.GENERAL_DIRECT_MATCH_TEMPLATE).format(pii=pii_descriptions),
                "attack_difficulty": 1
            })

        return prompts

