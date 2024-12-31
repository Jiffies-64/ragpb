from rouge_score import rouge_scorer


def calculate_rouge_l(hypothesis, reference):
    """
    Calculate ROUGE-L scores between a hypothesis and a reference.
    :param hypothesis: The predicted text.
    :param reference: The ground truth text.
    :return: ROUGE-L scores (Precision, Recall, F1-score).
    """
    # Create a ROUGE scorer instance, specifying only ROUGE-L calculation
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Calculate ROUGE-L scores
    scores = scorer.score(reference, hypothesis)
    # Return the Precision, Recall, and F1-score of ROUGE-L
    return scores['rougeL']