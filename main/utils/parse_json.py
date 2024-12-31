import json
import re


def clean_code_block_syntax(text: str) -> str:
    """
    Remove code block syntax (e.g., ```json, ```) from the text to ensure compatibility with JSON loading.
    :param text: The input text.
    :return: The text without code block syntax.
    """
    # Remove code block markers like ```json or ```
    text = re.sub(r"```(\w+)?", "", text)  # Match ```json or ```
    text = text.strip()  # Remove any leading/trailing whitespace
    return text


def parse_resp_to_json(response: str):
    """
    Load JSON data from a response string after cleaning any code block syntax.
    :param response: The input response string.
    :return: The parsed JSON data, or None if parsing fails.
    """
    cleaned_response = clean_code_block_syntax(response)
    # Use regex to try to match the part that conforms to the JSON format
    json_match = re.search(r"{.*}", cleaned_response, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Replace single quotes with double quotes, replace None with null, and try parsing again
            json_str = json_str.replace("None", "null")
            return json.loads(json_str)
    return None
