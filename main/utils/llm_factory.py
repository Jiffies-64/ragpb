import os
from openai import OpenAI


class LLMManager:
    class ModelEnum:
        QWEN_PLUS = "qwen-plus-1125"
        QWEN_TURBO = "qwen-turbo-1101"
        QWEN_LONG = "qwen-long"
        GPT4o = "gpt-4o"

    def __init__(self, model_enum=ModelEnum.QWEN_PLUS):
        """
        Initialize the LLMManager instance and obtain the corresponding language model client.
        :param model_enum: The model enum type to use, default is ModelEnum.QWEN_PLUS.
        """
        self.model_enum = model_enum
        self.client = self.get_llm_client(model_enum)

    def get_llm_client(self, model_enum):
        """
        Obtain the corresponding language model client based on the specified model enum.
        :param model_enum: The model enum type.
        :return: The corresponding OpenAI client object.
        """
        if model_enum in (self.ModelEnum.QWEN_PLUS, self.ModelEnum.QWEN_TURBO, self.ModelEnum.QWEN_LONG):
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("API key not found. Please set DASHSCOPE_API_KEY environment variable.")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif model_enum == self.ModelEnum.GPT4o:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key not found. Please set OPENAI_API_KEY environment variable.")
            base_url = "https://chatapi.onechats.top/v1/"
        else:
            raise ValueError(f"Unknown model_enum: {model_enum}")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return client

    def get_llm_output(self, prompt, system_content="You are a helpful assistant.", temperature=0.7):
        """
        Use the initialized model client to obtain the output content of the model based on the input prompt (returned as a JSON formatted string).
        :param prompt: The prompt text input to the model.
        :param temperature: The temperature parameter for the model.
        :param system_content: The system prompt content, default is "You are a helpful assistant.".
        :return: The JSON formatted string of the model output.
        """
        model_name = str(self.model_enum)
        if model_name.find('qwen')!= -1:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=temperature
            )
            return completion.model_dump()['choices'][0]['message']['content']
        elif model_name.find('gpt')!= -1:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=temperature
            )
            return completion.model_dump()['choices'][0]['message']['content']
        else:
            print('LLM client is disable.')
            exit(1)
