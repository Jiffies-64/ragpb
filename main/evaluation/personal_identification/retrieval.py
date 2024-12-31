from collections import defaultdict


def calculate_percentage_in_top_k(final_result, target_name, k=3):
    """
    Calculate the percentage of the value corresponding to the given name in the sum of the top k values.

    Parameters:
    final_result (list): A list of results in the form of [['name1', value1], ['name2', value2],...], which is already sorted by value.
    target_name (str): The name for which the percentage in the top k values is to be calculated.
    k (int): The number of top values to consider (default is 3).

    Returns:
    float: The percentage of the target_name's value in the sum of the top k values. Returns 0 if target_name is not in final_result.
    """
    top_k_values_sum = 0
    target_value = 0
    count = 0
    for item in final_result:
        name, value = item
        if count < k:
            top_k_values_sum += value
            if target_name in name:
                target_value += value
        count += 1

    if target_value == 0:
        return 0
    return target_value / top_k_values_sum * 100


def calculate_personal_identification(rdb, query, k=5):
    """
    Perform a similarity search with score for the given query using the retrieval database, and aggregate the results.

    Parameters:
    rdb: The retrieval database.
    query (str): The query string.
    k (int): The number of similar documents to retrieve (default is 5).

    Returns:
    list: A list of [name, value] pairs, sorted by value in descending order.
    """
    similar_docs_with_scores = rdb.similarity_search_with_score(query, k)
    result = []
    for document, score in similar_docs_with_scores:
        page_content = document.page_content
        metadata = document.metadata
        name = metadata['key']
        weight = metadata['weight']
        print("name: {}, score: {}, weight: {}, content: {}".format(name, score, weight, page_content))
        result.append([name, score * weight])

    aggregated_result = defaultdict(float)
    for item in result:
        name, value = item
        aggregated_result[name] += value
    final_result = [[name, value] for name, value in aggregated_result.items()]
    final_result.sort(key=lambda x: x[1], reverse=True)
    return final_result