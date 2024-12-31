import csv
import sys
import names
from dotenv import load_dotenv

from main.utils.llm_factory import LLMManager

PROCESS_TEMPLATE = """
==== Task Instructions ====
You are an expert in the medical field. Below is a set of synthetic medical dialogue data, which includes a patient's question and a response from ChatDoctor AI. I will also provide the patient's name. Your task is to:
1. Infer the patient's gender based on the context of their question or other clues from the dialogue and select the appropriate name (male or female).
2. Incorporate the patient's name, and if possible, infer additional relevant details (such as age or gender) based on the context, ensuring that this information is naturally integrated into the conversation and does not disrupt the logical flow.
3. Rephrase any references to 'ChatDoctor' or similar AI terms. Replace them with appropriate responses that a human doctor would give.
4. Ensure the doctor’s response is clear, concise, and consistent with prior dialogue, avoiding unnecessary details, complex explanations, or the introduction of new terminology.
5. Ensure the conversation sounds realistic and coherent, with no awkward or forced phrasing.

==== Example Input ====
Patient's name: John Doe or Jane Smith
Patient: “I’ve been feeling short of breath lately, especially when exercising. What could be the cause?”
Doctor: “ChatDoctor suggests that shortness of breath could be due to various factors, such as asthma or heart conditions.”
==== Example Output ====
Patient: “I’m a 35-year-old male, and I’ve been feeling short of breath lately, especially when exercising. What could be the cause?”; Doctor: “John, shortness of breath during exercise could be caused by several factors, including asthma, heart conditions, or even anxiety.”

==== Input ====
Patient's name: {} or {}
Patient: {}
Doctor: {}
==== Output format ====
Patient: {{Processed_Question}}; Doctor: {{Processed_Response}}
"""
generated_names = set()
load_dotenv()
llm_manager = LLMManager(LLMManager.ModelEnum.QWEN_TURBO)


def generate_unique_name(gender=None):
    while True:
        if gender == 'male':
            name = names.get_first_name(gender='male')
        elif gender == 'female':
            name = names.get_first_name(gender='female')
        else:
            name = names.get_first_name()

        if name not in generated_names:
            generated_names.add(name)
            return name


def _read_txt_file(txt_file_path):
    data_list = []
    input_line = None
    output_line = None
    with open(txt_file_path, "r", encoding="utf-8") as txt_file:
        for line in txt_file:
            line = line.strip()
            if not line:
                if input_line and output_line:
                    data_list.append((input_line, output_line))
                input_line = None
                output_line = None
            elif input_line is None:
                input_line = line.lstrip("input: ")
            else:
                output_line = line.lstrip("output: ")
        if input_line and output_line:
            data_list.append((input_line, output_line))
    return data_list


def write_to_csv(name, row, csv_file_path):
    with open(csv_file_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerow([name, row])


def process(raw_data):
    male = generate_unique_name('male')
    female = generate_unique_name('female')
    prompt = PROCESS_TEMPLATE.format(male, female, raw_data[0], raw_data[1])
    try:
        resp = llm_manager.get_llm_output(prompt)
        if male in resp:
            return male, resp
        else:
            return female, resp
    except Exception as e:
        print(f"An exception occurred while processing data {raw_data}: {e}", file=sys.stderr)
        return None, None


if __name__ == '__main__':
    start_line = 0
    CHATDOCTOR_TXT_PATH = "chatdoctor.txt"
    OUTPUT_PATH = "../chatdoctor-plus.csv"

    # Initialize the CSV with headers before starting the generation process
    if start_line == 0:
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            writer.writerow(["name", "text"])

    raw_data_list = _read_txt_file(CHATDOCTOR_TXT_PATH)
    raw_data_list = raw_data_list[start_line:]

    # Process each data point and write immediately to CSV
    for raw_data in raw_data_list:
        name, response = process(raw_data)
        if name is not None and response is not None:
            write_to_csv(name, response, OUTPUT_PATH)
