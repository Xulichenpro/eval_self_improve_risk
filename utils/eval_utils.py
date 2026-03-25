def jaccard_similarity(list1: set[str], list2: set[str]) -> float:
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union if union > 0 else 0.0