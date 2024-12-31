from json import JSONDecodeError
from typing import List, Optional, Dict, Any

from main.utils.parse_json import parse_resp_to_json


class Symptom:
    def __init__(self, symptom_name: str, symptom_details: str, symptom_duration: Optional[str] = None):
        """
        Initialize a Symptom object.

        Parameters:
        symptom_name (str): The name of the symptom.
        symptom_details (str): The details of the symptom.
        symptom_duration (str, optional): The duration of the symptom (default is None).
        """
        self.symptom_name = symptom_name
        self.symptom_details = symptom_details
        self.symptom_duration = symptom_duration

    def __repr__(self):
        """
        String representation of the Symptom object.

        Returns:
        str: A string representation of the Symptom object.
        """
        return f"Symptom(name={self.symptom_name}, details={self.symptom_details}, duration={self.symptom_duration})"

    def to_dict(self) -> Dict[str, Optional[str]]:
        """
        Convert the Symptom object to a dictionary.

        Returns:
        dict: A dictionary representing the Symptom object.
        """
        return {
            "symptom_name": self.symptom_name,
            "symptom_details": self.symptom_details,
            "symptom_duration": self.symptom_duration
        }


class PatientPrivacyInfo:
    def __init__(self, name: Optional[str], gender: Optional[str], age: Optional[str],
                 symptoms: List[Symptom], additional_privacy_attributes: Dict[str, Optional[str]]):
        """
        Initialize a PatientPrivacyInfo object.

        Parameters:
        name (str, optional): The patient's name.
        gender (str, optional): The patient's gender.
        age (str, optional): The patient's age.
        symptoms (List[Symptom]): A list of Symptom objects.
        additional_privacy_attributes (Dict[str, str], optional): Additional privacy attributes.
        """
        self.name = name
        self.gender = gender
        self.age = age
        self.symptoms = symptoms
        self.additional_privacy_attributes = additional_privacy_attributes
        self.known_symptom_names = []

    @classmethod
    def from_json_str(cls, data: str) -> "PatientPrivacyInfo":
        """
        Create a PatientPrivacyInfo object from a JSON string.

        Parameters:
        data (str): The JSON string.

        Returns:
        PatientPrivacyInfo: An instance of PatientPrivacyInfo.
        """
        try:
            data = parse_resp_to_json(data)
            # Validate and extract the main fields
            name = data.get("name", None)
            gender = data.get("gender", None)
            age = data.get("age", None)

            # Extract symptoms, handling missing or malformed symptom data
            symptoms_data = data.get("symptoms", [])
            symptoms = []
            for s in symptoms_data:
                symptom_name = s.get("symptom_name", "unknown symptom")
                symptom_details = s.get("symptom_details", "no details provided")
                symptom_duration = s.get("symptom_duration", None)
                symptoms.append(Symptom(symptom_name, symptom_details, symptom_duration))

            # Validate additional privacy attributes, setting defaults if missing
            additional_privacy_attributes = data.get("additional_privacy_attributes", {})
            additional_privacy_attributes = {
                "chronic_conditions": additional_privacy_attributes.get("chronic_conditions", None),
                "medications": additional_privacy_attributes.get("medications", None),
                "previous_procedures": additional_privacy_attributes.get("previous_procedures", None),
                "other_relevant_information": additional_privacy_attributes.get("other_relevant_information", None)
            }

            # Create and return an instance of PatientPrivacyInfo
            return cls(name, gender, age, symptoms, additional_privacy_attributes)

        except (TypeError, KeyError, AttributeError, JSONDecodeError) as e:
            # Log the error and return a default PatientPrivacyInfo instance
            print(f"Error while parsing data: {e}")
            return cls(
                name=None,
                gender=None,
                age=None,
                symptoms=[],
                additional_privacy_attributes={}
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the PatientPrivacyInfo object to a dictionary.

        Returns:
        dict: A dictionary representing the PatientPrivacyInfo object.
        """
        # Convert symptoms information to a list of dictionaries
        symptoms_list = [symptom.to_dict() for symptom in self.symptoms]
        return {
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "symptoms": symptoms_list,
            "additional_privacy_attributes": self.additional_privacy_attributes
        }

    def __repr__(self):
        """
        String representation of the PatientPrivacyInfo object.

        Returns:
        str: A string representation of the PatientPrivacyInfo object.
        """
        return (f"PatientPrivacyInfo(name={self.name}, gender={self.gender}, age={self.age}, "
                f"symptoms={self.symptoms}, additional_privacy_attributes={self.additional_privacy_attributes})")

    def get_unused_attributes(self, used_symptoms: List[Symptom]) -> Dict[str, Optional[str]]:
        """
        Identify attributes not used in the prompt to check for potential privacy leaks.

        Parameters:
        used_symptoms (List[Symptom]): A list of used Symptom objects.

        Returns:
        dict: A dictionary of unused attributes.
        """
        unused_attributes = {"name": self.name, "gender": self.gender, "age": self.age}

        # Symptoms that weren't included in the prompt
        unused_symptoms = [sym for sym in self.symptoms if sym not in used_symptoms]
        unused_attributes["unused_symptoms"] = [{"symptom_name": sym.symptom_name,
                                                 "symptom_details": sym.symptom_details,
                                                 "symptom_duration": sym.symptom_duration}
                                                for sym in unused_symptoms]

        # Additional privacy attributes that were not used
        unused_attributes["additional_privacy_attributes"] = self.additional_privacy_attributes

        return unused_attributes
