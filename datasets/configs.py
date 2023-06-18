logical_fallacy_labels_to_ids = {
    "ad populum": 0,
    "faulty generalization": 1,
    "fallacy of logic": 2,
    "false dilemma": 3,
    "appeal to emotion": 4,
    "fallacy of relevance": 5,
    "intentional": 6,
    "false causality": 7,
    "fallacy of credibility": 8,
    "ad hominem": 9,
    "circular reasoning": 10,
    "equivocation": 11,
    "fallacy of extension": 12,
}

dataset_to_max_length = {
    "imdb": 512,
    "dbpedia": 512,
    "ag_news": 64,
}

dataset_to_num_labels = {
    "imdb": 2,
    "dbpedia": 9,
    "ag_news": 4,
}
