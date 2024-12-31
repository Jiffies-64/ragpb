### Datasets 

Medical Dataset Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10364849/ 

Legal Dataset Link: https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus 



### Evaluation Metrics

![ragpb](README.assets/ragpb.png)



### Operation Guide

#### Create a virtual environment (optional)

```Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### Install dependencies

```Bash
pip install -r requirements.txt
```

#### Modify the .env file

Modify `TASK_ID` to be the id of a task. A task can be understood as testing a privacy protection method on a dataset. Therefore, it is recommended that `TASK_ID` contains both the name of the dataset and the name of the protection method, such as `legal_basic`.

Modify the API key according to the model to be used, such as `OPENAI_API_KEY`.

#### Build a vector database

This part will use the original dataset to build a raw text vector database and an atomic sentence vector database for subsequent retrieval.

```Bash
python scripts/step0_init_env.py --dataset chatdoctor-plus
```

#### Generate privacy attack prompts

This step will generate prompts that may cause privacy leaks according to certain rules, or use templates, or call large language models, and save them in the database. At the same time, two files will be generated in the root directory `output` folder. The `prompt` file records the id and specific prompts, and the `response template` records the id and the response column to be filled.

```Bash
python scripts/step1_before_attack.py --dataset chatdoctor-plus
```

#### Externally call the privacy protection method

Call the RAG privacy protection method to be evaluated to obtain the response and fill it back into the response template, and upload it to the `user_upload` folder under the root directory.

#### Evaluate adversarial samples

This step will read the uploaded response and calculate the selected evaluation metrics. The results will be saved in the database, and a report will be generated in the root directory `output`, containing the average scores of each metric.

```Bash
python scripts/step2_after_response.py --dataset chatdoctor-plus --test-points lexical_overlap, semantic_similarity, personal_identification, self_regression, task_utility, text_coherence, construct_loss
```