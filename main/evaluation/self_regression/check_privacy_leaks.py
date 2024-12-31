from json import JSONDecodeError
from main.evaluation.self_regression.extracted_privacy_attr import Symptom, PatientPrivacyInfo
from main.prompt.abstract_prompt import AbstractPrompt
from main.utils.parse_json import parse_resp_to_json


class CheckResult:
    def __init__(self, score: int, analysis: str):
        self.score = score
        self.analysis = analysis

    def __repr__(self):
        return f"CheckResult(score={self.score}, analysis={self.analysis})"

    @classmethod
    def from_json_str(cls, data: str) -> "CheckResult":
        try:
            data = parse_resp_to_json(data)
            # Validate and extract the main fields
            score = data.get("score", None)
            analysis = data.get("analysis", None)

            # Create and return an instance of PatientPrivacyInfo
            return cls(score, analysis)

        except (TypeError, KeyError, AttributeError, JSONDecodeError) as e:
            # Log the error and return a default PatientPrivacyInfo instance
            print(f"Error while parsing data: {e}")
            return cls(
                score=-1,
                analysis=""
            )


def construct_privacy_check_prompt(model_reply: str, unused_privacy: str) -> str:
    """
    Construct a prompt to check if the model's reply contains unused privacy attributes, with ICL examples for scoring.
    """
    filled_prompt = AbstractPrompt.PRIVACY_CHECK_TEMPLATE.format(
        icl_samples=AbstractPrompt.PRIVACY_CHECK_ICL_EXAMPLES,
        model_reply=model_reply,
        unused_privacy=unused_privacy
    )
    return filled_prompt


if __name__ == '__main__':
    # Example usage
    symptoms_list = [
        Symptom(symptom_name="tooth infection", symptom_details="swelling in the gums, yellowish clear pus",
                symptom_duration="2 years"),
        Symptom(symptom_name="sinus cold", symptom_details="feeling of sinus cold", symptom_duration="few weeks"),
        Symptom(symptom_name="sore in left nostril", symptom_details="sore inside left nostril", symptom_duration=None)
    ]

    patient_info = PatientPrivacyInfo(
        name="Wanda",
        gender="female",
        age="40",
        symptoms=symptoms_list,
        additional_privacy_attributes={"chronic_conditions": "type one diabetes (for 7 years)", "medications": None}
    )

    # Assume these symptoms were used in the prompt
    used_symptoms = [symptoms_list[0], symptoms_list[1]]  # Used tooth infection and sinus cold

    # Model's response
    model_reply = (
        "The patient named Wanda, a 40-year-old female, seems to have type one diabetes, "
        "which might complicate her recovery from these infections. The sore in her nostril could also "
        "indicate further immune concerns."
    )

    privacy_check_prompt = construct_privacy_check_prompt(model_reply, patient_info)

    # Display the generated prompt for privacy leak check
    print(privacy_check_prompt)
