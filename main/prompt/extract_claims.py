from langchain_core.prompts import PromptTemplate

CLAIMS_EXTRACTION_EXAMPLES = [
    {
        "Document": "Recent studies indicate that moderate exercise reduces the risk of heart disease, improves "
                    "mental health, and increases life expectancy. Additionally, a balanced diet is crucial in "
                    "maintaining overall health, especially in reducing the risk of chronic diseases such as diabetes "
                    "and obesity.",
        "Claims": [
            "Moderate exercise reduces the risk of heart disease.",
            "Moderate exercise improves mental health.",
            "Moderate exercise increases life expectancy.",
            "A balanced diet is crucial for maintaining overall health.",
            "A balanced diet helps reduce the risk of chronic diseases such as diabetes.",
            "A balanced diet helps reduce the risk of obesity."
        ]
    }
]

CLAIMS_EXTRACTION_EXAMPLES_BY_PERSON = [
    {
        "Document": "In the case of Smith v Johnson, the judge ruled that Mr. Smith had the right to claim damages "
                    "for the property damage caused by Mr. Johnson's negligence. Mrs. Brown, a witness, "
                    "testified that she saw Mr. Johnson speeding before the accident. Mr. Adams, the lawyer for Mr. "
                    "Smith, argued convincingly about the extent of the damages.",
        "Claims": """{
            "Mr. Smith": [
                "Has the right to claim damages for property damage caused by Mr. Johnson's negligence"
            ],
            "Mrs. Brown": [
                "Testified that she saw Mr. Johnson speeding before the accident"
            ],
            "Mr. Adams": [
                "The lawyer for Mr. Smith",
                "Argued convincingly about the extent of the damages"
            ]
        }"""
    }
]

CLAIMS_EXTRACTION_TEMPLATE = PromptTemplate(
    input_variables=["icl_samples", "document"],
    template="""
    Please breakdown the following input into a set of atomic, independent claims, and return each of the claim in a new line.

    -- samples --
    {icl_samples}

    -- document to breakdown --
    {document}

    Return claims with <SEP> intervals only.
    """
)

CLAIMS_EXTRACTION_TEMPLATE_BY_PERSON = PromptTemplate(
    input_variables=["icl_samples", "document"],
    template="""
Please breakdown the following input into a set of atomic, independent claims and return them in JSON format.

-- samples --
{icl_samples}

-- document to breakdown --
{document}

Return the claims in JSON format with keys as the names and values as lists of claims.
"""
)
