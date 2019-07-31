def compute_mean_avg_prec(top_doc_ids, train_df, test_df, label_type):

    """Computes Mean Average Precision over all queries based on the ranking of the collection
    top_doc_ids: dimension (number of queries, collection size), contains the ranked document IDs for
    each query
    train_df: source of collection, needed to retrieve ground truth labels in collection
    test_df: source of queries, needed to retrieve ground truth labels of queries
    label_type: type of class used for validation "PRODUCT" or "ISSUE"
    """

    total_avg_prec = 0

    for query_id, doc_ids in enumerate(top_doc_ids):
        correct = 0
        sum_prec = 0
        query_label = test_df.iloc[query_id][label_type]

        for i, id in enumerate(doc_ids):
            if train_df.iloc[id][label_type] == query_label:
                correct += 1
                prec = correct / (i + 1)
                sum_prec += prec
        relevant = sum([1 for label in list(train_df[label_type]) if label == query_label])
        if relevant != 0:
            avg_prec = sum_prec / relevant
            total_avg_prec += avg_prec

    mean_avg_prec = total_avg_prec / len(test_df)

    return mean_avg_prec

